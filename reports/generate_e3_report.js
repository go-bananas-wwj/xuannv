const { Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel } = require('docx');
const fs = require('fs');

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "SimSun", size: 24 }
      }
    },
    paragraphStyles: [
      {
        id: "Title",
        name: "Title",
        basedOn: "Normal",
        run: { size: 44, bold: true, color: "000000", font: "SimHei" },
        paragraph: { spacing: { before: 240, after: 240 }, alignment: AlignmentType.CENTER }
      },
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 32, bold: true, color: "000000", font: "SimHei" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 }
      },
      {
        id: "BodyText",
        name: "Body Text",
        basedOn: "Normal",
        run: { size: 24, font: "SimSun" },
        paragraph: { spacing: { line: 360, after: 120 }, indent: { firstLine: 480 } }
      }
    ]
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
    },
    children: [
      new Paragraph({
        heading: HeadingLevel.TITLE,
        children: [new TextRun("EarthEmbeddingExplorer年度标志性成果")]
      }),

      // 成果1
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("成果一：构建全球遥感Embedding跨模态检索平台，打破学术成果\"发完即藏\"的困境")]
      }),
      new Paragraph({
        style: "BodyText",
        children: [new TextRun("当前国际遥感领域发表了大量高影响力的基础模型，但用户需要下载数百GB数据、自行搭建环境、编写复杂代码才能使用，学术成果与实际应用之间存在巨大鸿沟。本项目集成DINOv2、FarSLIP、SatCLIP、SigLIP等国际代表性遥感基础模型，构建统一的跨模态检索平台，支持文本、图像、地理位置三种查询方式，覆盖全球约1.4%陆地表面的卫星影像。实现了从\"发表论文\"到\"实际可用\"的关键跨越，让遥感基础模型的能力不再局限于论文图表，而是转化为科研人员触手可得的实用工具。")]
      }),

      // 成果2
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("成果二：部署在线交互式网站，实现遥感AI零门槛使用")]
      }),
      new Paragraph({
        style: "BodyText",
        children: [new TextRun("基于ModelScope Studio完成云端部署，提供免费GPU运行环境。用户无需下载海量数据、无需编写一行代码、无需任何本地配置，打开网页即可进行全球遥感影像的跨模态检索与分析。这一部署模式将高门槛的遥感基础模型能力彻底普惠化，使科研人员、政府决策者乃至非专业用户都能便捷地使用国际最先进的遥感AI能力。网站的上线标志着遥感基础模型从\"专家工具\"向\"公共服务\"的转变，为全球地球科学研究、生态环境监测、自然资源管理等领域提供了开放、免费、即开即用的技术入口。")]
      }),

      // 成果3
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("成果三：建立遥感基础模型标准化对比评估体系，支撑领域科学共识形成")]
      }),
      new Paragraph({
        style: "BodyText",
        children: [new TextRun("不同遥感基础模型的训练数据、架构设计和评估方式各异，研究者难以横向比较，模型选择缺乏客观依据。本项目在同一平台、同一数据集、同一检索任务下，系统对比语义对齐、视觉特征、位置编码等不同类型模型的检索行为与地理偏差，直观揭示各模型在不同气候带、不同地物类型上的能力差异与局限性。为遥感基础模型的选型、改进和标准化评估提供了开放的对比工具和实证依据，推动领域从\"各自为战\"走向\"可对比、可复现、可迭代\"，促进了国际遥感AI社区形成科学共识，加速了模型性能的透明化与优化进程。")]
      })
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/workspace/xuannv/reports/EarthEmbeddingExplorer_年度标志性成果报告.docx", buffer);
  console.log("EarthEmbeddingExplorer报告已生成: /workspace/xuannv/reports/EarthEmbeddingExplorer_年度标志性成果报告.docx");
});
