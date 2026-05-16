const pptxgen = require('pptxgenjs');
const path = require('path');

const CHARTS_DIR = '/workspace/xuannv/references/charts';

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_16x9';
pptx.title = '玄女底座 — 模型+平台全栈汇报';
pptx.author = 'XuanNv Team';

// 统一风格
const TITLE_STYLE = { fontSize: 28, bold: true, color: '000000', fontFace: 'Arial' };
const SUBTITLE_STYLE = { fontSize: 16, color: '333333', fontFace: 'Arial' };
const BODY_STYLE = { fontSize: 14, color: '000000', fontFace: 'Arial' };
const NOTE_STYLE = { fontSize: 11, color: '666666', fontFace: 'Arial', italic: true };

function addImageSlide(title, imgFile, notes) {
    const slide = pptx.addSlide();
    slide.background = { color: 'FFFFFF' };
    
    // 标题
    slide.addText(title, {
        x: 0.5, y: 0.3, w: 9, h: 0.6,
        ...TITLE_STYLE
    });
    
    // 图片
    const imgPath = path.join(CHARTS_DIR, imgFile);
    const imgW = 9; // 最大宽度
    const imgH = 4.5; // 估算高度
    slide.addImage({
        path: imgPath,
        x: 0.5, y: 1.0,
        w: imgW, h: imgH,
        sizing: { type: 'contain', w: imgW, h: imgH }
    });
    
    // 底部注释
    if (notes) {
        slide.addText(notes, {
            x: 0.5, y: 5.7, w: 9, h: 0.5,
            ...NOTE_STYLE
        });
    }
    
    return slide;
}

// ========== P1: 封面 ==========
const s1 = pptx.addSlide();
s1.background = { color: 'FFFFFF' };
s1.addText('玄女底座', {
    x: 1, y: 2.0, w: 8, h: 1.2,
    fontSize: 48, bold: true, color: '000000', fontFace: 'Arial',
    align: 'center'
});
s1.addText('自研地球嵌入基座 + 可视化展示平台', {
    x: 1, y: 3.3, w: 8, h: 0.6,
    fontSize: 20, color: '333333', fontFace: 'Arial',
    align: 'center'
});
s1.addText('模型训练 · 下游推理 · 交互标注 · AI 智能体 — 全栈产品', {
    x: 1, y: 4.2, w: 8, h: 0.5,
    fontSize: 14, color: '666666', fontFace: 'Arial',
    align: 'center'
});

// ========== P2: 模型原理 ==========
addImageSlide('模型原理：输入 → STP 编码器 → VMF 瓶颈 → 64 维输出',
    '05_model_architecture.png',
    '训练时 Skip L2 防坍缩，推理时 L2 + VMF，时序对比是核心差异。');

// ========== P3: 核心亮点 ==========
const s3 = pptx.addSlide();
s3.background = { color: 'FFFFFF' };
s3.addText('核心亮点', { x: 0.5, y: 0.3, w: 9, h: 0.6, ...TITLE_STYLE });

const highlights = [
    ['01  反坍缩训练', 'Uniformity 为负，嵌入均匀散开。第一次训练崩到 -0.1，调了十几轮才稳住。'],
    ['02  时序敏感', '不重叠双窗口训练，月度级变化可检测。AEF 用年度数据，我们在月度 granularity。'],
    ['03  纯自研', '代码、结构、训练流程全部自己写的。知识产权干净，升级不受制于人。']
];

highlights.forEach(([title, desc], i) => {
    const y = 1.2 + i * 1.5;
    s3.addText(title, { x: 0.8, y, w: 8.5, h: 0.5, fontSize: 18, bold: true, color: '000000', fontFace: 'Arial' });
    s3.addText(desc, { x: 0.8, y: y + 0.5, w: 8.5, h: 0.6, ...BODY_STYLE });
});

// ========== P4: 变化检测 ==========
addImageSlide('变化检测：BA 79.8%（含 CD Head）/ 50%（Raw）',
    '01_cd_comparison_ba.png',
    '加 CD Head 超 AEF 是事实；不加时 50% 也是事实，V6 在补。');

// ========== P5: 展示平台总览 ==========
const s5 = pptx.addSlide();
s5.background = { color: 'FFFFFF' };
s5.addText('展示平台总览', { x: 0.5, y: 0.3, w: 9, h: 0.6, ...TITLE_STYLE });

s5.addText('不是 PPT，是真能点、能切、能出报告的网页。', {
    x: 0.5, y: 1.0, w: 9, h: 0.4, fontSize: 16, color: '1a5276', fontFace: 'Arial', bold: true
});

const modules = [
    ['三维地球可视化', 'Three.js 全球训练样本分布，缩放到 patch 级别看详情'],
    ['下游监测能力', '5 大 Task Head 动态切换，Mosaic 大图拼接，Patch 详情弹窗'],
    ['AI 智能体报告', '自然语言输入，DeepSeek-V4 解析 → 执行 → 润色出报告']
];
modules.forEach(([name, desc], i) => {
    const y = 1.6 + i * 1.3;
    s5.addText(name, { x: 0.8, y, w: 3, h: 0.4, fontSize: 16, bold: true, color: '000000', fontFace: 'Arial' });
    s5.addText(desc, { x: 4.0, y, w: 5.5, h: 0.5, ...BODY_STYLE });
});

s5.addText('技术栈：React 19 · TypeScript · Three.js · MapLibre · FastAPI · Docker',
    { x: 0.5, y: 5.5, w: 9, h: 0.4, ...NOTE_STYLE, align: 'center' });

// ========== P6: 五大下游任务 ==========
addImageSlide('五大下游任务：全部已上线',
    '08_downstream_tasks.png',
    '4 / 5 任务纯 CPU 推理，单 patch ~10 ms。CD Head 自动选择空闲 GPU。');

// ========== P7: 交互式标注训练 ==========
addImageSlide('交互式标注训练：零代码自定义训练',
    '09_annotation_pipeline.png',
    '业务人员自己就能训专用分类模型；传统方式至少两周，现在 5 分钟。');

// ========== P8: AI 智能体报告 ==========
addImageSlide('AI 智能体报告：DeepSeek-V4 驱动',
    '10_agent_report.png',
    '已经接通，不是规划。输一句话就能出带表格和分析的报告。');

// ========== P9: 轻量化 ==========
addImageSlide('轻量化：23 TB → 3 GB → 16 MB',
    '06_compression_ladder.png',
    '16 MB 比表情包还小；平台上所有推理底层都是这 64 字节。');

// ========== P10: 效率提升 ==========
addImageSlide('效率提升 vs 传统遥感分析',
    '04_efficiency_comparison.png',
    '数字来自实际项目；存储降是因为下游不需要原始影像。');

// ========== P11: 升级路线 ==========
addImageSlide('升级路线：V5 → V6 → V6.5 → V7',
    '07_roadmap.png',
    'V5 已验证，V6 在训，V6.5 已设计，V7 长期目标。');

// ========== P12: 总结 ==========
const s12 = pptx.addSlide();
s12.background = { color: 'FFFFFF' };
s12.addText('总结', { x: 0.5, y: 0.3, w: 9, h: 0.6, ...TITLE_STYLE });

const summary = [
    ['够轻', '64 字节 / km²，23 TB 压到 16 MB，任何设备离线可用'],
    ['够敏', '月度级时序敏感，变化检测 BA 79.8%'],
    ['够全', '模型 + 平台 + 5 大任务 + 交互训练 + AI 智能体，一整套产品'],
    ['够诚', '从零自研，优势不夸大，短板不回避，路线清晰']
];

summary.forEach(([word, desc], i) => {
    const y = 1.2 + i * 1.2;
    s12.addText(word, { x: 1.0, y, w: 1.5, h: 0.5, fontSize: 24, bold: true, color: '1a5276', fontFace: 'Arial' });
    s12.addText(desc, { x: 3.0, y: y + 0.1, w: 6, h: 0.5, ...BODY_STYLE });
});

s12.addText('有问题随时打断。', {
    x: 0.5, y: 5.5, w: 9, h: 0.4, fontSize: 14, color: '666666', fontFace: 'Arial',
    align: 'center', italic: true
});

// 保存
const outputPath = '/workspace/xuannv/references/玄女底座汇报_v5.pptx';
pptx.writeFile({ fileName: outputPath })
    .then(() => console.log(`PPT saved to: ${outputPath}`))
    .catch(err => { console.error('Error:', err); process.exit(1); });
