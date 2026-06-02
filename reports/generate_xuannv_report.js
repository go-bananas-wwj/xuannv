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
        children: [new TextRun("玄女底座年度标志性成果")]
      }),

      // 成果1
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("成果一：构建AI-Ready地球嵌入模型，一个底座支撑多下游任务")]
      }),
      new Paragraph({
        style: "BodyText",
        children: [new TextRun("模型输出的通用地理嵌入向量可直接用于变化检测、地物分类、空间异常检测、土地覆盖分析等多个下游任务。传统遥感AI开发需要针对每个业务场景从头训练专用模型，周期长、成本高、泛化能力弱。本项目实现了\"一个预训练底座+灵活下游适配\"的新范式，各业务团队无需重复训练主干网络，仅需在预训练嵌入向量上轻量微调或接入简单检测头，即可快速上线定制化应用。这一模式大幅降低了遥感AI应用的开发门槛和计算成本，使科研机构、行业单位能够聚焦业务逻辑而非模型训练本身，显著加速了从技术研发到业务落地的转化效率。")]
      }),

      // 成果2
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("成果二：突破国际现有模型瓶颈，实现月度级时间敏感性与反坍缩双重升级")]
      }),
      new Paragraph({
        style: "BodyText",
        children: [new TextRun("国际主流的地球基础模型存在两大核心瓶颈：一是嵌入向量坍缩，不同地点、不同时相的表征趋于雷同；二是对时间变化不敏感，且国际上变化检测普遍以年度为尺度，难以满足月度乃至更短周期的精细监测需求。本项目通过独创的反坍缩机制与时序对比学习，使嵌入向量在保持空间判别力的同时，在时间维度上产生显著差异，将变化检测能力从年度尺度下探到月度尺度，从无效提升至可用水平，使模型真正\"看懂\"地球表面的时间演变。在国际上首次在有限数据条件下实现了地球基础模型的时间敏感性突破，为耕地非农化监测、冰川演变追踪、滑坡灾害预警等时序任务提供了全新的技术工具，填补了该领域的国际空白。")]
      }),

      // 成果3
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("成果三：部署在线可视化平台，实现遥感变化检测能力的直观展示与交互验证")]
      }),
      new Paragraph({
        style: "BodyText",
        children: [new TextRun("建设了交互式Web可视化平台，支持嵌入空间分布展示、全域变化强度热力图、空间异常检测、训练过程监控等功能。用户可在界面上自主选择时间窗口，实时生成变化检测结果，实现\"指哪看哪、选时对比\"的交互体验。该平台将技术能力从实验室转化为可感知、可演示的产品形态，为业务落地和客户展示提供了直接抓手。领导与专家无需理解复杂模型原理，即可通过直观的可视化界面快速验证模型能力；业务人员可以基于平台开展初步的变化筛查，大幅缩短从技术研发到业务应用的衔接周期，为遥感AI成果的规模化推广奠定了展示基础。")]
      })
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/workspace/xuannv/reports/玄女底座_年度标志性成果报告.docx", buffer);
  console.log("玄女底座报告已生成: /workspace/xuannv/reports/玄女底座_年度标志性成果报告.docx");
});
