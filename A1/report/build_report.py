"""Build report figures, HTML, and PDF from saved A1 artifacts.

Run --figures with A1's scientific Python environment. Run --export with a
Python environment containing reportlab, Pillow, and pypdf. Neither operation
trains models or changes experiment artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
A1 = HERE.parent
MODELS = ['linear', 'mlp', 'cnn', 'lstm', 'transformer']
LABELS = ['Linear', 'MLP', 'CNN', 'LSTM', 'Transformer']


def load_runs():
    runs = []
    for name in MODELS:
        folder = A1 / 'runs' / ('m1' if name in ['linear', 'mlp'] else 'm2') / name
        read = lambda f: json.loads((folder / f).read_text(encoding='utf-8'))
        with (folder / 'history.csv').open(newline='') as f:
            history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
        runs.append({'name': name, 'folder': folder, 'summary': read('summary.json'),
                     'test': read('test_summary.json'), 'verification': read('verification.json'),
                     'history': history})
    return runs


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    out = HERE / 'assets'
    out.mkdir(exist_ok=True)
    runs = load_runs()
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    colors = ['#526577', '#3d7c91', '#277b51', '#99682e', '#875684']
    x = np.arange(5)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)
    for a, key, title, low, fmt in zip(ax, ['test_accuracy', 'test_macro_f1'],
                                      ['Official test accuracy', 'Official test macro-F1'],
                                      [0, 0], ['.2f', '.4f']):
        values = [r['test'][key] * (100 if key == 'test_accuracy' else 1) for r in runs]
        bars = a.bar(x, values, color=colors, width=.64)
        a.set_xticks(x, LABELS)
        a.tick_params(axis='x', labelsize=10)
        a.set_ylim(0, 105 if key == 'test_accuracy' else 1.05)
        a.set_ylabel('Accuracy (%)' if key == 'test_accuracy' else 'Macro-F1')
        a.set_title(title, fontweight='bold', fontsize=12)
        a.bar_label(bars, labels=[format(v, fmt) for v in values], padding=4, fontsize=10)
        a.yaxis.grid(True, alpha=.18)
        a.set_axisbelow(True)
    fig.savefig(out / 'test_comparison.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for i, (r, a) in enumerate(zip(runs, axes.flat)):
        h = r['history']; epochs = [v['epoch'] for v in h]
        a.plot(epochs, [v['train_loss'] for v in h], '--', color='#355d86', label='Training')
        a.plot(epochs, [v['val_loss'] for v in h], '-', color='#ad572c', label='Validation')
        a.axvline(r['summary']['best_epoch'], color='#777', alpha=.55, linestyle=':', label='Selected epoch')
        a.set_title(LABELS[i], fontweight='bold')
        a.set_xlabel('Epoch'); a.set_ylabel('Cross-entropy')
        a.set_xticks([1, 3, 5, 7, 10]); a.grid(alpha=.18)
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='center', frameon=False, fontsize=12)
    axes.flat[-1].text(.5, .2, 'Separate y-axis ranges\nShared seed and ten-epoch budget',
                       ha='center', transform=axes.flat[-1].transAxes, fontsize=10)
    fig.savefig(out / 'learning_comparison.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for i, r in enumerate(runs):
        acc = 100 * r['test']['test_accuracy']
        axes[0].scatter(r['summary']['parameters'], acc, color=colors[i], s=65)
        axes[1].scatter(r['test']['inference']['forward_batch_ms'], acc, color=colors[i], s=65)
        for j, a in enumerate(axes):
            px = r['summary']['parameters'] if j == 0 else r['test']['inference']['forward_batch_ms']
            offset = (-8, 8) if r['name'] == 'transformer' else (7, 7)
            a.annotate(LABELS[i], (px, acc), xytext=offset, textcoords='offset points',
                       ha='right' if r['name'] == 'transformer' else 'left', fontsize=10)
    axes[0].set_xscale('log'); axes[0].set_xlabel('Trainable parameters (log scale)')
    axes[0].set_title('Model size and accuracy', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Forward latency for batch of 128 (ms)')
    axes[1].set_title('Recorded CUDA forward latency and accuracy', fontweight='bold', fontsize=12)
    axes[1].set_xlim(-.08, 2.35)
    for a in axes:
        a.set_ylabel('Test accuracy (%)'); a.set_ylim(82.5, 94); a.grid(alpha=.2)
    fig.savefig(out / 'efficiency_comparison.png', dpi=200)
    plt.close(fig)


def blocks(source):
    """Parse the deliberately small Markdown subset used by this report."""
    lines = source.splitlines(); i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1; continue
        if line.startswith('```'):
            code = []; i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]); i += 1
            yield ('code', '\n'.join(code)); i += 1; continue
        match = re.match(r'^(#{1,3}) (.+)$', line)
        if match:
            yield ('heading', (len(match[1]), match[2])); i += 1; continue
        match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
        if match:
            yield ('image', (match[1], match[2])); i += 1; continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            yield ('table', rows); continue
        if re.match(r'^\d+\. ', line):
            items=[]
            while i<len(lines) and re.match(r'^\d+\. ', lines[i]):
                items.append(re.sub(r'^\d+\. ', '', lines[i])); i+=1
            yield ('list',items); continue
        para = [line]; i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '```', '|', '![')):
            para.append(lines[i]); i += 1
        yield ('paragraph', '\n'.join(para))


def inline(s, pdf=False):
    s = html.escape(s)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
               lambda m: '<a href="' + m[2] + '">' + m[1] + '</a>', s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', s)
    s = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>' if pdf else r'<code>\1</code>', s)
    return s.replace('  \n', '<br/>').replace('\n', ' ')


def export_html(parsed):
    parts = []
    nav = []
    for kind, value in parsed:
        if kind == 'heading':
            level, title = value
            anchor = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
            parts.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
            if level == 2: nav.append(f'<a href="#{anchor}">{html.escape(title)}</a>')
        elif kind == 'paragraph':
            parts.append('<p>' + inline(value) + '</p>')
        elif kind == 'code':
            parts.append('<pre><code>' + html.escape(value) + '</code></pre>')
        elif kind == 'list':
            parts.append('<ol>' + ''.join('<li>'+inline(v)+'</li>' for v in value) + '</ol>')
        elif kind == 'table':
            rows = []
            for j, cells in enumerate(value):
                tag = 'th' if j == 0 else 'td'
                rows.append('<tr>' + ''.join(f'<{tag}>{inline(c)}</{tag}>' for c in cells) + '</tr>')
            parts.append('<div class="table-wrap"><table><thead>' + rows[0] +
                         '</thead><tbody>' + ''.join(rows[1:]) + '</tbody></table></div>')
        elif kind == 'image':
            caption, path = value
            parts.append(f'<figure><img src="{html.escape(path)}" alt="{html.escape(caption)}" loading="lazy">'
                         f'<figcaption>{html.escape(caption)}</figcaption></figure>')
    css = '''
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#eceff2;color:#171a1e;
    font-family:Georgia,"Times New Roman",serif;font-size:17px;line-height:1.65}main{max-width:1000px;
    margin:32px auto;background:white;padding:58px 68px}h1,h2,h3{color:#000;line-height:1.25;
    font-family:Arial,sans-serif}h1{font-size:34px;margin:0 0 25px}h2{font-size:25px;margin:48px 0 20px;
    scroll-margin-top:24px}h3{font-size:19px;margin:28px 0 12px}p{margin:0 0 16px}a{color:#22567b}
    table{border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:13px;line-height:1.45;
    margin:12px 0 24px}td,th{border:1px solid #d9d9d9;padding:10px 9px;text-align:right;vertical-align:middle}
    td:first-child,th:first-child{text-align:left}th{background:#e9eff4;color:#000}tbody tr:nth-child(even){background:#f7f9fb}
    .table-wrap{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.55;
    background:#f4f5f6;padding:16px 20px;border:1px solid #ddd}code{font-size:.83em;font-family:Consolas,monospace}
    figure{margin:26px 0 30px;break-inside:avoid}figure img{display:block;max-width:100%;height:auto;margin:auto}
    figcaption{font-size:13px;text-align:center;color:#414951;margin-top:10px;font-family:Arial,sans-serif}
    @media(max-width:700px){main{margin:0;padding:28px 20px}body{font-size:16px}h1{font-size:27px}}
    @media print{body{background:white;font-size:10.5pt;line-height:1.45}main{max-width:none;margin:0;padding:0}
    h1{font-size:24pt}h2{font-size:16pt;break-after:avoid}h3{font-size:12pt;break-after:avoid}
    a{color:inherit;text-decoration:none}table{font-size:8pt}td,th{padding:6px}figure img{max-height:160mm}
    thead{display:table-header-group}tr{break-inside:avoid}@page{size:A4;margin:20mm}}
    '''
    result = '<!doctype html><html lang="en"><head><meta charset="utf-8">' + \
             '<meta name="viewport" content="width=device-width, initial-scale=1">' + \
             '<title>Assignment 1 Fashion MNIST Report</title><style>' + css + '</style></head>' + \
             '<body><main>' + '\n'.join(parts) + '</main></body></html>'
    (HERE / 'report.html').write_text(result, encoding='utf-8')


def export_pdf(parsed):
    from PIL import Image as PILImage
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image, KeepTogether, PageBreak, Preformatted)
    from reportlab.platypus.tableofcontents import TableOfContents

    fonts = Path('C:/Windows/Fonts')
    for name, file in [('ReportSerif','times.ttf'), ('ReportSerif-Bold','timesbd.ttf'),
                       ('ReportSerif-Italic','timesi.ttf'), ('ReportSerif-BoldItalic','timesbi.ttf'),
                       ('ReportSans','arial.ttf'), ('ReportSans-Bold','arialbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily('ReportSerif', normal='ReportSerif', bold='ReportSerif-Bold',
                                  italic='ReportSerif-Italic', boldItalic='ReportSerif-BoldItalic')
    pdfmetrics.registerFontFamily('ReportSans', normal='ReportSans', bold='ReportSans-Bold',
                                  italic='ReportSans', boldItalic='ReportSans-Bold')
    W, H = A4; margin = 52; width = W - 2*margin
    base = ParagraphStyle('Body', fontName='ReportSerif', fontSize=11, leading=15,
                          spaceAfter=9, allowWidows=0, allowOrphans=0)
    styles = {
      1: ParagraphStyle('Title', parent=base, fontName='ReportSans-Bold', fontSize=24, leading=29, spaceAfter=23),
      2: ParagraphStyle('H2', parent=base, fontName='ReportSans-Bold', fontSize=16, leading=20,
                        spaceBefore=19, spaceAfter=10, keepWithNext=True),
      3: ParagraphStyle('H3', parent=base, fontName='ReportSans-Bold', fontSize=12, leading=16,
                        spaceBefore=12, spaceAfter=7, keepWithNext=True),
    }
    caption = ParagraphStyle('Caption', parent=base, fontName='ReportSans', fontSize=9,
                             leading=12, alignment=TA_CENTER, spaceAfter=12)
    table_caption = ParagraphStyle('TableCaption', parent=base, fontName='ReportSans',
                                   fontSize=9, leading=12, keepWithNext=True, spaceAfter=5)
    cell = ParagraphStyle('Cell', parent=base, fontName='ReportSans', fontSize=8.5,
                          leading=11, spaceAfter=0, alignment=TA_CENTER)
    first_cell = ParagraphStyle('FirstCell', parent=cell, alignment=TA_LEFT)
    code_style = ParagraphStyle('Code', fontName='Courier', fontSize=7.5, leading=10,
                                spaceBefore=4, spaceAfter=11, leftIndent=8,
                                backColor=colors.HexColor('#f4f5f6'), borderPadding=8)
    toc=TableOfContents()
    toc.levelStyles=[ParagraphStyle('TOCEntry',parent=base,fontSize=11,leading=17,
                                    spaceBefore=10,spaceAfter=2)]
    toc.dotsMinLevel=0
    story = []
    for kind, value in parsed:
        if kind == 'heading':
            level, title = value
            if level == 2 and title in ['Abstract', 'Contents', '1 Introduction',
                                       '10 References', 'Appendix A Reproduction instructions',
                                       'Appendix C Additional confusion matrices']:
                story.append(PageBreak())
            heading=Paragraph(inline(title, True), styles[level])
            if level==2 and title!='Contents': heading._toc_title=title
            story.append(heading)
            if title=='Contents': story.append(toc)
        elif kind == 'paragraph':
            s = table_caption if value.startswith('**Table ') else base
            story.append(Paragraph(inline(value, True), s))
        elif kind == 'code':
            wrapped = []
            for line in value.splitlines():
                wrapped.extend(textwrap.wrap(line, width=94, replace_whitespace=False,
                                             drop_whitespace=False, subsequent_indent='    ') or [''])
            story.append(Preformatted('\n'.join(wrapped), code_style))
        elif kind == 'list':
            # The report's sole ordered list is replaced by the paginated TOC.
            continue
        elif kind == 'image':
            cap, relative = value; path = (HERE / relative).resolve()
            iw, ih = PILImage.open(path).size
            scale = min(width / iw, 310 / ih)
            pic = Image(str(path), width=iw*scale, height=ih*scale)
            story.append(KeepTogether([Spacer(1, 8), pic, Spacer(1, 6), Paragraph(html.escape(cap), caption)]))
        elif kind == 'table':
            n = len(value[0])
            if n == 2: ratios = [.42, .58]
            elif n == 3: ratios = [.30, .43, .27]
            elif n == 4: ratios = [.24, .22, .22, .32]
            elif n == 5: ratios = [.23, .16, .19, .21, .21]
            elif n == 6: ratios = [.25, .15, .15, .15, .15, .15]
            else: ratios = [1/n]*n
            if len(ratios) != n: ratios = [1/n]*n
            rows = [[Paragraph(inline(c, True), first_cell if j == 0 else cell)
                     for j,c in enumerate(row)] for row in value]
            table = Table(rows, colWidths=[width*r for r in ratios], repeatRows=1, hAlign='LEFT')
            commands = [('GRID',(0,0),(-1,-1),.4,colors.HexColor('#d9d9d9')),
                        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf3')),
                        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]
            for row in range(2,len(rows),2):
                commands.append(('BACKGROUND',(0,row),(-1,row),colors.HexColor('#f7f9fb')))
            table.setStyle(TableStyle(commands)); story.extend([table, Spacer(1, 12)])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('ReportSans', 8)
        canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(margin, 28, 'CO3133  |  Assignment 1  |  Fashion MNIST')
        canvas.drawRightString(W-margin, 28, str(doc.page))
        canvas.restoreState()
    output = HERE / 'assignment1_report.pdf'
    class ReportDocument(SimpleDocTemplate):
        def afterFlowable(self,flowable):
            title=getattr(flowable,'_toc_title',None)
            if title:
                key='section-'+re.sub(r'[^a-z0-9]+','-',title.lower()).strip('-')
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(title,key,0)
                self.notify('TOCEntry',(0,title,self.page,key))
    doc = ReportDocument(str(output), pagesize=A4, rightMargin=margin, leftMargin=margin,
                            topMargin=48, bottomMargin=48,
                            title='Fashion MNIST Classification with Five Neural Network Architectures',
                            author='Le Phuoc Vu; Nguyen Vu Long; Nguyen Hoang Quoc')
    doc.multiBuild(story, onFirstPage=footer, onLaterPages=footer)
    return output


def check_artifacts(parsed, pdf_path):
    from pypdf import PdfReader
    runs = load_runs(); evidence = []
    for r in runs:
        p = r['folder']; s=r['summary']; t=r['test']
        with (p/'test_predictions.csv').open(newline='') as f:
            pred=list(csv.DictReader(f))
        counts=[[0]*10 for _ in range(10)]
        for row in pred: counts[int(row['true_label'])][int(row['predicted_label'])] += 1
        accuracy=sum(counts[i][i] for i in range(10))/len(pred)
        f1=sum(2*counts[i][i]/(sum(counts[i])+sum(row[i] for row in counts)) for i in range(10))/10
        assert len(pred)==10000 and abs(accuracy-t['test_accuracy'])<1e-12
        assert abs(f1-t['test_macro_f1'])<1e-12
        assert min(r['history'],key=lambda row:row['val_loss'])['epoch']==s['best_epoch']
        with (p/'confusion_matrix.csv').open(newline='') as f:
            saved=list(csv.reader(f))[1:]
        assert counts==[[int(v) for v in row[1:]] for row in saved]
        assert r['verification']['status']=='passed'
        evidence.append({'model':r['name'],'accuracy':accuracy,'macro_f1':f1,
                         'selected_epoch':s['best_epoch'], 'rows':len(pred),
                         'test_summary_sha256':hashlib.sha256((p/'test_summary.json').read_bytes()).hexdigest()})
    images=[]
    for kind,value in parsed:
        if kind=='image':
            path=(HERE/value[1]).resolve(); assert path.is_file(),path
            from PIL import Image
            with Image.open(path) as image: image.verify()
            images.append(value[1])
    reader=PdfReader(str(pdf_path))
    text='\n'.join(p.extract_text() for p in reader.pages)
    assert len(reader.pages)>10
    for expected in ['91.94%', '88.50%', '88.48%', '85.55%', '84.26%', 'Appendix D', 'Lê Phước Vũ']:
        assert expected in text, expected
    missing_glyphs=[(i+1) for i,p in enumerate(reader.pages) if '\ufffd' in p.extract_text()]
    assert not missing_glyphs
    results={'pdf_pages':len(reader.pages),'source_words':len((HERE/'report.md').read_text(encoding='utf-8').split()),
             'figures':len(images),'images_valid':True,'models':evidence,
             'checks':'Prediction CSV accuracy/F1, saved confusion counts, checkpoint epoch, PDF text and image integrity',
             'ai_usage_sha256':hashlib.sha256((A1.parent/'AI_USAGE.md').read_bytes()).hexdigest()}
    (HERE/'report_checks.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',action='store_true')
    parser.add_argument('--export',action='store_true')
    args=parser.parse_args()
    if args.figures: figures()
    if args.export:
        parsed=list(blocks((HERE/'report.md').read_text(encoding='utf-8')))
        export_html(parsed)
        check_artifacts(parsed,export_pdf(parsed))


if __name__=='__main__': main()
