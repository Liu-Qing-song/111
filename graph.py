from graphviz import Digraph
import os


# 确保 Graphviz 可执行文件在系统路径中
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

def generate_flowchart_fixed():
    # 核心修改点：fontname="Microsoft YaHei"
    # 这能保证标题、子图标签、节点文字都支持中文
    dot = Digraph('Experiment_Design', comment='Signal Processing Experiment')

    # 1. 设置全局属性 (修复乱码的关键)
    dot.attr(fontname='Microsoft YaHei')  # 设置图表标题/子图标题的字体
    dot.attr(rankdir='TB', size='10', newrank='true')

    # 2. 设置节点和边属性
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='white', fontname='Microsoft YaHei')
    dot.attr('edge', fontname='Microsoft YaHei')

    # --- 阶段一 ---
    with dot.subgraph(name='cluster_0') as c:
        # 设置子图标签字体
        c.attr(fontname='Microsoft YaHei')
        c.attr(label='阶段一：信号建模 (Signal Modeling)', style='dashed')

        c.node('Input', '输入: 覆冰图像 f(x,y)\n(二维离散序列)')
        c.node('FFT', '2D-FFT 频谱分析')
        c.node('Spectrum', '频谱特征:\n低频(背景) vs 高频(边缘+冰刺)')

        c.edge('Input', 'FFT')
        c.edge('FFT', 'Spectrum')

    # --- 阶段二 ---
    with dot.subgraph(name='cluster_1') as c:
        c.attr(fontname='Microsoft YaHei')
        c.attr(label='阶段二：系统响应对比 (System Response)', style='dashed')

        c.node('LTI', '线性系统 (LTI)\n高斯滤波 (卷积)', fillcolor='#e1f5fe')
        c.node('NonLinear', '非线性系统\n中值滤波 (排序统计)', fillcolor='#e8f5e9')
        c.node('Res1', '响应 A:\n边缘模糊 (带限失真)')
        c.node('Res2', '响应 B:\n保边去噪 (去除脉冲噪声)')
        c.node('Select', '教学结论:\n选用非线性系统输出')

        c.edge('LTI', 'Res1')
        c.edge('NonLinear', 'Res2')
        c.edge('Res1', 'Select', style='dotted')
        c.edge('Res2', 'Select')

    # --- 阶段三 ---
    with dot.subgraph(name='cluster_2') as c:
        c.attr(fontname='Microsoft YaHei')
        c.attr(label='阶段三：特征提取 (Feature Extraction)', style='dashed')

        c.node('Diff', '差分系统 / 高通滤波\ny[n] = x[n] - x[n-1]')
        c.node('Grad', '梯度幅值 G\n(提取冲激响应)')

        c.edge('Diff', 'Grad')

    # --- 阶段四 ---
    with dot.subgraph(name='cluster_3') as c:
        c.attr(fontname='Microsoft YaHei')
        c.attr(label='阶段四：参数估计 (Parameter Estimation)', style='dashed')

        c.node('Rotate', '系统辨识: 坐标旋转校正')
        c.node('Slice', '降维处理: 一维切片分析')
        c.node('Measure', '物理参数反演:\n测量脉冲间隔 -> 冰厚', shape='ellipse', style='filled', fillcolor='#fff9c4')

        c.edge('Rotate', 'Slice')
        c.edge('Slice', 'Measure')

    # 连接各阶段
    dot.edge('Spectrum', 'LTI')
    dot.edge('Spectrum', 'NonLinear')
    dot.edge('Select', 'Diff')
    dot.edge('Grad', 'Rotate')

    # 保存并渲染
    output_filename = 'experiment_flowchart_fixed'
    dot.render(output_filename, view=True, format='png', cleanup=True)
    print(f"修复后的流程图已生成: {output_filename}.png")


if __name__ == '__main__':
    generate_flowchart_fixed()