import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
from collections import Counter

import os
import re
import math
import pandas as pd
# import seaborn as sns
from pyvenn import venn
# from functools import reduce
# from matplotlib.patches import Ellipse, Circle
# import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# from networkx.algorithms import community
# from networkx.algorithms import bipartite
# import pandas as pd
import seaborn as sns
# from adjustText import adjust_text

from adjustText import adjust_text
# from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
# from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from matplotlib.patches import Ellipse, Circle
import matplotlib.transforms as transforms
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.axisartist as axisartist
import plotly.express as px
import plotly.graph_objects as go
from tq_info import tools
from tq_info.tools import RGB_to_Hex

def enterdir(path):
    path0 = os.getcwd()
    path = path0 + path
    isexit = os.path.exists(path)
    if not isexit:
        os.mkdir(path)
        os.chdir(path)
    else:
        os.chdir(path)
    return


def outdir():
    os.chdir("../")


def _pheatmap(data, fx_r=None, fx_c=None, colour=["green", "black", "red"],heatmap_name="heatmap",**kwargs):
    '''函数接受两个参数；
    以行名索引的dataframe， 和保存图片的名称'''
    importr('pheatmap',lib_loc = "C:/Users/Pioneer/Documents/R/win-library/3.6")
    importr("base")
    shape = data.shape

    # data=np.log2(data)
    col = [x for x in data.columns]
    data = pandas2ri.py2ri(data)
    data._set_colnames(col)
    if shape[1] <= 4:
        scale = "column"
    else:
        scale = "row"

    if fx_r is None:
        fx_r1 = (580 / shape[0]) * 0.75
        fx_r2 = 580 / shape[0] * 0.55

    else:
        fx_r1 = fx_r2 = fx_r

    if fx_c is None:
        fx_c = 230 / shape[1] * 0.55


    def draw():
        if shape[0] >= 25:
            robjects.r["pheatmap"](data,
                     treeheight_row=30,
                     treeheight_col=30,
                     scale=scale,
                     cluster_cols=robjects.r['F'],
                     cluster_rows=robjects.r['T'],
                     display_numbers= robjects.r["F"],
                     number_format="%.3f",
                     fontsize_row=fx_r1,
                     fontsize_col=fx_c,
                     cellwidth=230 / shape[1], cellheight=580 / shape[0],
                     show_colnames=robjects.r['T'],
                     color=robjects.r['colorRampPalette'](robjects.r['c'](colour[0],colour[1],colour[2]),bias=1)(300),border_color=r('NA'))
        else:
            robjects.r["pheatmap"](data,
                       treeheight_row=30,
                       treeheight_col=30,
                       scale=scale,
                       cluster_cols=robjects.r['F'],
                       cluster_rows=robjects.r['F'],
                       display_numbers=robjects.r["F"],
                       number_format="%.3f",
                       fontsize_row=fx_r2,
                       fontsize_col=fx_c,
                       cellwidth=230 / shape[1], cellheight=380 / shape[0],
                       show_colnames=robjects.r['T'],
                       color=robjects.r['colorRampPalette'](robjects.r['c'](colour[0],colour[1],colour[2]), bias=1)(
                           300), border_color=r('NA'))
        return

    enterdir("/heatmap")
    robjects.r['pdf']("{heatmap_name}.pdf".format(heatmap_name=heatmap_name), width=8, height=12)
    draw()
    robjects.r["dev.off"]()
    robjects.r['png']("{heatmap_name}.png".format(heatmap_name=heatmap_name), width=576, height=864)
    draw()
    robjects.r["dev.off"]()
    outdir()
    return


def _volcano(x, y, r, logx=True, name="volcano",colour=["r","black","limegreen"]):
    """参数解释：
    x : FC可迭代对象
    y ：P值可迭代对象
    logx : 是否对FC取对数
    name : 输出图形名称"""

    mma = max([abs(np.log2(np.array([x for x in x if x != 0]).min())),
         np.log2(np.array([x for x in x if x != np.inf]).max())]) * 10
    x = [x if x != np.inf else mma for x in x]
    x = [x if x != 0 else 1/mma for x in x]

    if logx is True:
        x = np.log2([x for x in x])
    else:
        pass
    y = -np.log10([y for y in y])
    x_up, y_up = [], []
    x_down, y_down = [], []
    x_n, y_n = [], []
    for f, p in zip(x, y):
        if f >= np.log2(r) and p > -np.log10(0.05):
            x_up.append(f)
            y_up.append(p)
        elif f <= np.log2(1 / r) and p > -np.log10(0.05):
            x_down.append(f)
            y_down.append(p)
        else:
            x_n.append(f)
            y_n.append(p)

    fig = plt.figure(figsize=(5.6, 5.6))
    ax = fig.add_subplot(1, 1, 1)

    mx = np.max([abs(x) for x in x_up + x_down]) + 0.1
    if mx < 2 * r - 1:
        mx = 2 * r - 1

    print(colour[0],colour[1],colour[2],colour)
    my = np.max([x for x in [y_up + y_down + y_n] if str(x) != "nan"])
    ax.scatter(x_up, y_up, color=colour[0], s=15,label="up")
    ax.scatter(x_down, y_down, color=colour[2], s=15,label="down")
    ax.scatter(x_n, y_n, color=colour[1], s=15)

    ax.axhline(y=1.3, linewidth=1, color='black', linestyle='--')
    ax.axvline(x=-np.log2(r), linewidth=1, color='black', linestyle='--')
    ax.axvline(x=np.log2(r), linewidth=1, color='black', linestyle='--')

    ax.text(0.55 * mx, -np.log10(0.05) - 0.3 * (my / 8.5), s="P-value=0.05")
    ax.text(x=np.log2(r), y=my * 0.92, s="FC={r}".format(r=r))
    ax.text(x=-np.log2(r) - 2 * mx * 0.15, y=my * 0.92, s="FC=1/{rv}".format(rv=r))
    #ax.text(x=-np.log2(r) - 2 * mx * 0.12, y=my * 0.92, s="FC=0.8")
    ax.set_xlabel("log2(FC)")
    ax.set_ylabel("-log10(P-value)")
    ax.set_title("Volcano Plot")
    ax.legend(loc=None)

    plt.xlim([-mx, mx])
    enterdir("/volcano")
    plt.savefig('{compare}.pdf'.format(compare=name))
    plt.savefig('{compare}.png'.format(compare=name))
    outdir()
    return


def _veenx(labels=None, names=None,fontsize=None):
    """接受两个列表参数
    labels: 元素列表
    names: 标签列表"""
    labs = venn.get_labels(labels, fill=["number"])
    enterdir("/venn")
    if len(labels) == 2:
        fig, ax = venn.venn2(labs, names=names,fontsize=fontsize)
        fn = "&".join(names)
        plt.savefig("venn-{fn}.png".format(fn=fn))
        plt.savefig("venn-{fn}.pdf".format(fn=fn))
    if len(labels) == 3:
        fig, ax = venn.venn3(labs, names=names,fontsize=fontsize)
        fn = "&".join(names)
        plt.savefig("venn-{fn}.png".format(fn=fn))
        plt.savefig("venn-{fn}.pdf".format(fn=fn))
    if len(labels) == 4:
        fig, ax = venn.venn4(labs, names=names,fontsize=fontsize)
        fn = "&".join(names)
        plt.savefig("venn-{fn}.png".format(fn=fn))
        plt.savefig("venn-{fn}.pdf".format(fn=fn))
    if len(labels) == 5:
        fig, ax = venn.venn5(labs, names=names,fontsize=fontsize)
        fn = "&".join(names)
        plt.savefig("venn-{fn}.png".format(fn=fn))
        plt.savefig("venn-{fn}.pdf".format(fn=fn))
    outdir()
    return


def _flower(label,numbel):
    """接受两个参数
    numbers: 元素数量列表
    label: 标签列表"""
    fig, ax = plt.subplots(figsize=(10, 10))
    n = len(label)
    c = 0.35
    angle = 360 / n
    color = sns.hls_palette(n, l=.3, s=.8)
    for i in range(0, n):
        anx = angle * i
        ell = Ellipse(xy=(0.0 + c * math.cos(anx / 180 * math.pi),
                          0.0 + c * math.sin(anx / 180 * math.pi)),
                      width=1, height=0.05 + 2.8 / n,
                      angle=anx, facecolor=color[i], alpha=0.3)
        if len(label) >= 13:
            if 90 < anx < 270:
                plt.text(0.0 + 1.05 * math.cos(anx / 180 * math.pi),
                         0.0 + 1.05 * math.sin(anx / 180 * math.pi),
                         label[i], ha='center', va='center', rotation=anx + 180)
            else:
                plt.text(0.0 + 1.05 * math.cos(anx / 180 * math.pi),
                         0.0 + 1.05 * math.sin(anx / 180 * math.pi),
                         label[i], ha='center', va='center', rotation=anx)

        else:
            plt.text(0.0 + 1 * math.cos(anx / 180 * math.pi),
                     0.0 + 1 * math.sin(anx / 180 * math.pi),
                     label[i], ha='center', va='top')

        plt.text(0.0 + 0.6 * math.cos(anx / 180 * math.pi),
                 0.0 + 0.6 * math.sin(anx / 180 * math.pi),
                 numbel[i], ha='center', va='center')

        ax.add_patch(ell)

    cirle = Circle(xy=(0.0, 0.0), radius=0.2, facecolor='orangered', alpha=1)
    plt.text(0.0, 0.0, numbel[-1], ha='center', va='center')
    ax.add_patch(cirle)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('off')
    fn = "&".join(label)
    plt.savefig("venn-{fn}.png".format(fn=fn))
    plt.savefig("venn-{fn}.pdf".format(fn=fn))
    return


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def _pca(data=None, group=None, ellp=None, dim=True, figname=False,fontsize=10, s=120,lay=None,sign=None,html=None,ncol=None,**kwars):
    """
    Create a plot of the pca

    Parameters
    ----------
    data: dataframe, input data

    group: dict, group of sample

    ellp: circle | ellp | both ,style of ellipse

    dim: True | False, default is True,if False plot pca2 only

    figname: str, name of plot

    fontsize: int, fontsize

    s: int, size of point

    lay: float,(0 - 1)  percent of legend in the plot

    sign: True | False  if False will not sign the text,default True

    ncol: int, number of the legend columns"""

    # 固定顺序
    # row_order = reduce(lambda x,y:dic_g[x]+dic_g[y], dic_g)
    enterdir("/PCA")

    row_order = []
    dic_g = group
    dic_gc = {}
    for g, i in zip(group, range(len(group))):
        dic_gc[g] = i

    lx = [x for x in group]
    for key in dic_g:
        row_order += dic_g[key]
    data = data.loc[:, row_order]

    # 提取列名标签
    # labels = data.columns
    if figname is None:
        name1 = "PCA-" + "_".join([x for x in dic_g]) + "_2D"
        name2 = "PCA-" + "_".join([x for x in dic_g]) + "_3D"
        name3 = "PCA-" + "_".join([x for x in dic_g])
    else:
        name1 = "PCA-all_2D"
        name2 = "PCA-all_3D"
        name3 = "PCA-all"

    data = data.T
    data = preprocessing.scale(data)
    pca3 = PCA(n_components=3)
    d3 = pca3.fit_transform(data)
    l_x, l_y, l_z, l_g, l_s = [], [], [], [], []
    dic_cmap = {}
    for i in d3:
        l_x.append(-i[0])
        l_y.append(-i[1])
        l_z.append(-i[2])
    evr = pca3.explained_variance_ratio_

    # 二维图
    sns.set_style("white")
    fig, ax = plt.subplots()
    plt.rcParams['font.sans-serif'] = 'Arial'
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots_adjust(left=None, bottom=None, right=lay, top=None, wspace=None, hspace=None)
    ax.set_title('PCA')
    marker = ["o","D", "s", "X", "+", "^",  "p", "*", "h", "v", "8", "d", "H"]
    marker = ["o", "D", "s", "^", "p", "*", "h", "X", "+", "v", "8", "d", "H"]
    color = sns.color_palette("bright")

    if ellp is "circle":
        confidence_ellipse(np.array(l_x), np.array(l_y), ax, edgecolor='black', **kwars)
    elif ellp is "ellp":
        n = len(group[next(iter(group))])
        confidence_ellipse(np.array(l_x[0:n]), np.array(l_y[0:n]), ax, edgecolor=color[dic_gc[lx[0]]%10], facecolor=color[dic_gc[lx[0]]%10], alpha=0.4)
        confidence_ellipse(np.array(l_x[n:]), np.array(l_y[n:]), ax, edgecolor=color[dic_gc[lx[1]]%10],facecolor=color[dic_gc[lx[1]]%10], alpha=0.4)
    elif ellp is "both":
        confidence_ellipse(np.array(l_x), np.array(l_y), ax, edgecolor='black')
        n = len(group[next(iter(group))])
        confidence_ellipse(np.array(l_x[0:n]), np.array(l_y[0:n]), ax, edgecolor='black')
        confidence_ellipse(np.array(l_x[n:]), np.array(l_y[n:]), ax, edgecolor='black')

    i, j = 0, 0
    for x in dic_g:
        # ax.scatter(l_x[j: j + len(dic_g[x])], l_y[j: j + len(dic_g[x])],
        #            marker=marker[i % 11], c=[color[i % 10]], s=s, label=x)
        ax.scatter(l_x[j: j + len(dic_g[x])], l_y[j: j + len(dic_g[x])],
                   marker=marker[dic_gc[x]%11], c=[color[dic_gc[x]%10]], s=s, label=x)
        dic_cmap[x] = color[dic_gc[x]%10]
        i += 1
        j += len(dic_g[x])

    if sign is True:
        texts = [plt.text(m, n, k, fontsize=fontsize) for m, n, k in zip(l_x, l_y, row_order)]
        adjust_text(texts)
    else:
        pass
    ax.set_xlabel('pc1: %.2f%%' % (evr[0] * 100))
    ax.set_ylabel('pc2: %.2f%%' % (evr[1] * 100))
    plt.axvline(0, color="black", lw=1, zorder=0.5)
    plt.axhline(0, color="black", lw=1, zorder=0.5)
    plt.legend(bbox_to_anchor=(1,1),edgecolor="white",fontsize=fontsize,ncol=ncol,labelspacing=1.1)
    if lay is None:
        plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig("{name}.png".format(name=name1))
    plt.savefig("{name}.pdf".format(name=name1))

    if dim is True:
        fig = plt.figure()
        plt.subplots_adjust(left=None, bottom=None, right=lay, top=None, wspace=None, hspace=None)
        ax2 = fig.add_subplot(111, projection='3d')

        i, j = 0, 0
        for x in dic_g:
            ax2.scatter(l_x[j: j + len(dic_g[x])], l_y[j: j + len(dic_g[x])], l_z[j:j + len(dic_g[x])],
                        marker=marker[dic_gc[x]%11], c=[color[dic_gc[x]%10]], s=s, label=x)
            xx=marker[dic_gc[x] % 11]
            print(xx,5555555555555555)
            i += 1
            j += len(dic_g[x])
            l_g += [x] * len(dic_g[x])
            l_s += dic_g[x]
        if sign is True:
            texts = [ax2.text(m, n, z, k) for m, n, z, k in zip(l_x, l_y, l_z, row_order)]
            adjust_text(texts)
        plt.legend(bbox_to_anchor=(1.15,1),edgecolor="white",**kwars)
        if lay is None:
            plt.tight_layout()
        ax2.set_xlabel('pc1: %.2f%%' % (evr[0] * 100))
        ax2.set_ylabel('pc2: %.2f%%' % (evr[1] * 100))
        ax2.set_zlabel('pc3: %.2f%%' % (evr[2] * 100))
        plt.savefig("{name}.png".format(name=name2))
        plt.savefig("{name}.pdf".format(name=name2))

    if html is True:
        data = pd.DataFrame([l_x,l_y,l_z,l_g,l_s])

        data = data.T
        data.columns=["pc1","pc2","pc3","pg","ps"]

        rx = (data["pc1"].max() - data["pc1"].min()) * 0.15
        ry = (data["pc2"].max() - data["pc2"].min()) * 0.15
        rz = (data["pc3"].max() - data["pc3"].min()) * 0.15
        range_x = (data["pc1"].min()-rx, data["pc1"].max()+rx)
        range_y = (data["pc2"].min() - ry, data["pc2"].max() + ry)
        range_z = (data["pc3"].min() - rz, data["pc3"].max() + rz)

        def RGB_to_Hex(tmp):
            rgb = [x*255 for x in tmp]  # 将RGB格式划分开来
            # print(rgb)
            strs = '#'
            for i in rgb:
                num = int(i)  # 将str转int
                strs += str(hex(num))[-2:].replace('x', '0').upper()
            return strs

        dic_cmap2 = {}
        for k in dic_cmap:
            dic_cmap2[k] = RGB_to_Hex(dic_cmap[k])

        symbol = []
        for x in dic_g:
            symbol += marker[dic_gc[x] % 11]*len(dic_g[x])
            pass

        fig = px.scatter_3d(data, x='pc1', y='pc2', z='pc3', range_x=range_x, range_y=range_y, range_z=range_z,
                             color="pg", symbol="pg", color_discrete_map=dic_cmap2,hover_name="ps")
        l_legent = []
        for i in dic_g:
            l_legent.append({"name":i})
        fig.update(data=l_legent)

        fig.update_layout(scene={"aspectmode": "cube"})

        fig.write_html("{name}.html".format(name=name3))
    outdir()
    return


def _ppi(dproscore, epro, epath=None, dpath=None, dprofc=None, dprogene=None, figname="circular_layout",
        layout="circular_layout", adjust=True):
    """
       Create a plot of ppi

    Parameters
    ----------
    dpath: dict, the dict of pathway and p-value

    dprofc: dict, Optional parameters,dict of protein and fc,default is None

    dproscore: dict,  the score of protein

    epath: list, connection of path and protein

    epro: list, connection of protein and protein

    figname: string, Optional parameters

    layout: string, layout of plot"""

    fig, ax = plt.subplots(figsize=(10, 10))

    G = nx.Graph()

    if dpath is not None:
        dpath = sorted(dpath.items(), key=lambda x: len(x[0]), reverse=True)
        # 添加通路节点,和p值属性
        for path in dpath:
            G.add_node(path[0], pvalue=path[1])

    # 添加蛋白/基因节点,记录标准化后的得分
    dpro = sorted(dproscore.items(), key=lambda x: x[1])
    s = np.array([x[1] for x in dpro])
    s = ((s - s.min()) / (s.max() - s.min())) * 380 + 100

    # 添加蛋白/基因节点
    for pro, score in zip(dpro, s):
        G.add_node(pro[0], score=score)

    if layout == "circular_layout":
        nodePos = nx.circular_layout(G)
    elif layout == "spring_layout":
        nodePos = nx.spring_layout(G)

    # 绘制通路节点
    if dpath is not None:
        nx.draw_networkx_nodes(G, nodePos, nodelist=[x[0] for x in dpath], node_shape="s",
                               node_color=[x[1] for x in dpath], cmap=plt.get_cmap('cool'))

    # 分别定义上下颜色映射
    color_up = [(0.5, 0, 0), (1, 0, 0)]
    cmup = colors.ListedColormap(color_up, 'indexed')

    color_dowm = [(0, 0.5, 0), (0, 1, 0)]
    cmdown = colors.ListedColormap(color_dowm, 'indexed')

    # 绘制节点
    if dprofc is not None:
        uplist = [x for x in dprofc if float(dprofc[x]) > 1]
        upcolor = [dprofc[x] for x in uplist]
        s1 = np.array([G.node.get(x)["score"] for x in uplist])

        downlist = [x for x in dprofc if float(dprofc[x]) < 1]
        downcolor = [dprofc[x] for x in downlist]
        s2 = np.array([G.node.get(x)["score"] for x in downlist])

        nx.draw_networkx_nodes(G, nodePos, nodelist=uplist, node_shape="o",
                               node_color=upcolor,
                               cmap=plt.get_cmap(cmup), node_size=s1)

        nx.draw_networkx_nodes(G, nodePos, nodelist=downlist, node_shape="o",
                               node_color=downcolor,
                               cmap=plt.get_cmap(cmdown), node_size=s2)


    else:
        s0 = np.array([x[1] for x in dpro])
        s0 = ((s - s.min()) / (s.max() - s.min())) * 380 + 100
        nx.draw_networkx_nodes(G, nodePos, nodelist=[x[0] for x in dpro], node_shape="o",
                               node_color="grey",
                               node_size=s0)
    if dprogene is None:
        if adjust is True:
            texts = [plt.text(j[0], j[1], i, fontsize=12) for i, j in nodePos.items()]
            adjust_text(texts)
        else:
            for i, j in nodePos.items():
                plt.text(j[0], j[1] - 0.08, i, horizontalalignment='center')

    if dprogene is not None:
        if adjust is True:
            texts = [plt.text(j[0], j[1], dprogene[i], fontsize=12) for i, j in nodePos.items()]
            adjust_text(texts)
        else:
            for i, j in nodePos.items():
                plt.text(j[0], j[1] - 0.08, dprogene[i], horizontalalignment='center')

    # 绘制边
    if epath is not None:
        nx.draw_networkx_edges(G, nodePos, edgelist=epath, alpha=0.2, style="--")
    nx.draw_networkx_edges(G, nodePos, edgelist=epro, alpha=0.2, style="-")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig("{name}.png".format(name=figname))
    plt.savefig("{name}.pdf".format(name=figname))
    return


def _foldchang_bar1(s, name="foldchange_bars",ylab="Number Of Proteins"):
    """接受两个参数
    s: 列表，上下调个数 [100,-100]
    name : 输出图形的名称"""
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    plt.subplots_adjust(left=0.2, bottom=None, right=0.6, top=None, wspace=None, hspace=None)
    # plt.bar([1,2], s,width=0.55,color=["red", "green"])
    plt.bar([1], s[0], width=0.5, color="#f58200", label="up")
    plt.bar([2], s[1], width=0.5, color="#5c7a29", label="down")
    plt.xlim([0.25, 3])
    plt.tick_params(axis="y", which="major", width=3, length=8, labelsize=20)

    my = int(np.max([abs(x) for x in plt.ylim()]))

    plt.text(1, s[0] + my * 0.02, s[0], fontsize=20, ha='center')
    plt.text(2, s[1] - my * 0.08, abs(s[1]), fontsize=20, ha='center')
    ticks = ax.get_yticks()
    l = [int(abs(x)) for x in ticks]
    ax.set_yticklabels(l)
    ax.spines['bottom'].set_position(('data', 0))
    plt.xticks([])
    plt.ylabel(ylab, fontsize=25, labelpad=16)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=20, edgecolor="white")
    plt.savefig("{file}.png".format(file=name))
    plt.savefig("{file}.pdf".format(file=name))
    return


def _foldchang_bars(up, down, labx, bin=0.8, fx=None,ft=None, fy=15, name="foldchange_bars", ylab="Number Of Proteins"):
    """接受两个参数
    s: 列表，上下调个数 [100,-100]
    name : 输出图形的名称"""
    fig, ax = plt.subplots()
    #    fig.set_size_inches(12, 8)
    #    plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.8, wspace=None, hspace=None)

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    if fx is None:
        fx = fontsize=-len(up) + 18
    if ft is None:
        ft = -len(up) + 18

    for i, u in zip(range(len(up)), up):
        plt.bar(i - bin / 4, u, width=bin / 2, color="#f58200", label="up")

    for j, d in zip(range(len(up)), down):
        plt.bar(j + bin / 4, d, width=bin / 2, color="#5c7a29", label="down")
    #    plt.xlim([0.25, 3])

    plt.tick_params(axis="y", which="major", width=1.5, length=5, labelsize=fy)

    my = int(np.max([abs(x) for x in plt.ylim()]))

    for i, u in zip(range(len(up)), up):
        plt.text(i - bin / 4, u + my * 0.01, u, fontsize=ft, ha='center')
    for j, d in zip(range(len(up)), down):
        plt.text(j + bin / 4, d + my * 0.01, d, fontsize=ft, ha='center')

    plt.xticks(range(len(up)))
    ax.set_xticklabels(labx, fontsize=fx, rotation=45,ha='right')

    plt.ylabel(ylab, fontsize=fy, labelpad=16)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[len(up) - 1:len(up) + 1], labels=labels[len(up) - 1:len(up) + 1],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=fy,
              handlelength=1, edgecolor="white")

    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("{file}.png".format(file=name))
    plt.savefig("{file}.pdf".format(file=name))
    return


def _boxpplot(data,name=None):
    col = data.columns
    group = []
    value = []
    for i in col:
        value += data[i].values.tolist()
        group += [i]*len(data[i].values)
    da = pd.DataFrame([value,group]).T
    da.columns = ["value","group"]

    fig, ax = plt.subplots()
    color = sns.color_palette("bright")
    sns.boxplot(x="group", y="value", palette=color, data=da)
    labx = [re.search("'(.*)'", str(i)).group(1) for i in ax.get_xticklabels()]
    ax.set_xticklabels(labx, rotation=45, ha='right')
    #ax.spines['right'].set_visible(False)
    plt.savefig("boxplot-{name}.pdf".format(name=name))
    plt.savefig("boxplot-{name}.png".format(name=name))
    pass


class keggplot:

    def __init__(self):
        pass

    def paopao(self):
        if "KEGG.top.Total.xls" in os.listdir():
            paopao_script = """
            print(.libPaths())
            library("ggplot2")
            auto_bubble0<-function(data=data){
            pp = ggplot2::ggplot(data,aes(Enrichment_score,term))
            pbubble = pp + ggplot2::geom_point(aes(size=Count,color=pvalue)) +
            ggplot2::theme(axis.text=element_text(size=10),axis.title=element_text(size=10),legend.text=element_text(size=10),legend.title=element_text(size=10))
            pbubble + ggplot2::scale_colour_gradient(low="green",high="red")+guides(size = guide_legend(order = 1))
            ggsave("KEGG.top.Total.pdf",width = 7, height =5.5)
             ggsave("KEGG.top.Total.png",width = 7, height =5.5)
            }
            pathway<- read.csv("KEGG.top.Total.xls",sep="\t",check.names=F)
            auto_bubble0(data=pathway)
            """
            robjects.r(paopao_script)

        if "KEGG.top.Up.xls" in os.listdir():
            paopao_script = """
            library("ggplot2")
            auto_bubble0<-function(data=data){
            pp = ggplot2::ggplot(data,aes(Enrichment_score,term))
            pbubble = pp + ggplot2::geom_point(aes(size=Count,color=pvalue)) +
            ggplot2::theme(axis.text=element_text(size=10),axis.title=element_text(size=10),legend.text=element_text(size=10),legend.title=element_text(size=10))
            pbubble + ggplot2::scale_colour_gradient(low="green",high="red")+guides(size = guide_legend(order = 1))
            ggsave("KEGG.top.Up.pdf",width = 7, height =5.5)
             ggsave("KEGG.top.Up.png",width = 7, height =5.5)
            }
            pathway<- read.csv("KEGG.top.Up.xls",sep="\t",check.names=F)
            auto_bubble0(data=pathway)
            """
            robjects.r(paopao_script)

        if "KEGG.top.Down.xls" in os.listdir():
            paopao_script = """
            library("ggplot2")
            auto_bubble0<-function(data=data){
            pp = ggplot2::ggplot(data,aes(Enrichment_score,term))
            pbubble = pp + ggplot2::geom_point(aes(size=Count,color=pvalue)) +
            ggplot2::theme(axis.text=element_text(size=10),axis.title=element_text(size=10),legend.text=element_text(size=10),legend.title=element_text(size=10))
            pbubble + ggplot2::scale_colour_gradient(low="green",high="red")+guides(size = guide_legend(order = 1))
            ggsave("KEGG.top.Down.pdf",width = 7, height =5.5)
             ggsave("KEGG.top.Down.png",width = 7, height =5.5)
            }
            pathway<- read.csv("KEGG.top.Down.xls",sep="\t",check.names=F)
            auto_bubble0(data=pathway)
            """
            robjects.r(paopao_script)

        return

    def pathterm(self,term=None, pvalue=None, name="kegg"):
        sns.set_style("white")
        fig, ax = plt.subplots()
        color = sns.color_palette("GnBu_d", 10)
        d = -np.log(pvalue)
        vals = np.ones((10, 4))
        vals[:, 0], vals[:, 1] = 0.25, 0.25
        vals[:, 2] = [0.25 + (i / 9 * d[9] / d[i]) * 0.7 for i in range(10)]
        ax.bar(term, -np.log10(pvalue), color=vals)
        ax.set_xticks([x for x in range(10)])
        ax.set_xticklabels(term, rotation=45, ha="right")

        my = -np.log10(np.max(pvalue))
        ax.axhline(y=-np.log10(0.01), linewidth=0.5, color='red', linestyle='--')
        ax.text(10, -np.log10(0.01), s="P-value=0.01")

        ax.axhline(y=-np.log10(0.05), linewidth=0.5, color='blue', linestyle='--')
        ax.text(10, -np.log10(0.05) - 0.2 * my, s="P-value=0.05")
        ax.set_ylabel("-log10(P-value)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
        plt.savefig("{kegg}_pvalue.png".format(kegg=name))
        return

    def contrasterm(self,data=None, x=None, y=None, hue="type", palette=None, main=None, name=None):
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(6, 6.5))
        fig.set_size_inches(8, 6)
        plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None)

        ax = sns.barplot(x=x, y=y, hue=hue, palette=palette, data=data)
        plt.tick_params(labelsize=7)
        plt.ylabel("")
        plt.legend(loc=(1.04,0.5))
        #plt.legend(bbox_to_anchor=(1.04,0.5), loc="ceter left")
        plt.title("KEGG Pathway Classification({main})".format(main=main),  loc="left")
        plt.tight_layout()
        #plt.legend(bbox_to_anchor=(1.25, .5))
        plt.savefig("{file}.png".format(file=name))
        plt.savefig("{file}.pdf".format(file=name))
        return


class goplot:

    def __init__(self):
        pass

    def topgo(self,data=None,par=None):
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(8, 6))

        xlab = data["term_x"].tolist()

        bp = data[data["Classification_level1"] == "biological_process"]
        x1 = len(bp["term_x"].tolist())
        y1 = -np.log10(bp["pvalue"].tolist())

        cc = data[data["Classification_level1"] == "cellular_component"]
        x2 = len(cc["term_x"].tolist())
        y2 = -np.log10(cc["pvalue"].tolist())

        mf = data[data["Classification_level1"] == "molecular_function"]
        x3 = len(mf["term_x"].tolist())
        y3 = -np.log10(mf["pvalue"].tolist())

        s = x1 + x2 + x3
        print(x1,x2,x3)
        print([x for x in range(x1)],y1)
        print([x for x in range(x1,x2)], y2)

        plt.bar([x for x in range(x1)], y1, label="biological_process")
        plt.bar([x for x in range(x1,x1+x2)], y2,label="cellular_component")
        plt.bar([x for x in range(x1+x2,x1+x2+x3)], y3,label="molecular_function")
        plt.ylabel("-log10(pvalue)")
        plt.legend( bbox_to_anchor=(-0.048, 1.0), fontsize=10, edgecolor="white",handlelength=1)
        plt.xticks(range(x1+x2+x3))
        ax.set_xticklabels(xlab,fontsize=8, rotation=45,ha='right')
        plt.tight_layout()
        plt.savefig("GO.top.{par}.png".format(par=par))
        plt.savefig("GO.top.{par}.pdf".format(par=par))
        pass

    def allgobar(self,bpnum=None, bpterm=None, ccnum=None, ccterm=None, mfnum=None, mfterm=None,ylab="Number of genes"):
        b = bpnum + [0]
        c = ccnum + [0]
        m = mfnum

        l = b + c + m
        ln = bpterm + [" "] + ccterm + [" "] + mfterm

        print(b,c,m)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.bar([x for x in range(0, 11)], b)
        plt.bar([x for x in range(11, 22)], c)
        plt.bar([x for x in range(22, 32)], m)

        my = np.max((ax.get_yticks()))
        y = [my, my]
        x, x2, x3 = [-1, 10.5], [10.5, 21.5], [21.5, 32]

        plt.plot(x, y, lw=8)
        plt.plot(x2, y, lw=8)
        plt.plot(x3, y, lw=8)

        plt.text(5, my * 1.05, "Biological Process", ha="center")
        plt.text(16, my * 1.05, "Cell Component", ha="center")
        plt.text(27, my * 1.05, "Molecular Function", ha="center")

        ax.set_xticklabels(ln, rotation=30, fontsize=8)
        plt.xticks([x for x in range(0, 32)], ln, rotation=45, ha="right")
        plt.ylabel(ylab)

        y2 = plt.twinx()
        y2.set_yticklabels(["0", "20%", "40%", "60%", "80%"])
        y2.set_ylabel("percent of genes")
        ax.spines['top'].set_visible(False)
        y2.spines['top'].set_visible(False)
        plt.xlim([-1, 32])
        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, hspace=None, wspace=None)
        plt.savefig("all_go_bar.pdf")
        return

    def contrastonto(self,data=None, x=None, y=None,  palette=["red", "black"], name=None):
        sns.set(style="whitegrid")
        fig, ax = plt.subplots()
        fig.set_size_inches(12,6)
        bin = 1
        up= [x for x in data["diff-Total"]]
        down= [x for x in data["Unigene"]]
        labx= data["GO_classify2"]

        for i, u in zip(range(len(up)), up):
            plt.bar(i - bin / 4, u, width=bin / 2, color="#f58200", label="all")

        for j, d in zip(range(len(up)), down):
            plt.bar(j + bin / 4, d, width=bin / 2, color="#5c7a29", label="dep")

        plt.xticks(range(len(up)))

        ax.set_xticklabels(labx,rotation=90,fontsize=7,ha='right')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[len(up) - 1:len(up) + 1], labels=labels[len(up) - 1:len(up) + 1],
                  loc="upper left",  fontsize=7,
                  handlelength=1, edgecolor="white")
        plt.plot([1, 5], [-10, -10])
        # plt.text(1,-50,"-"*10)

        # plt.legend(bbox_to_anchor=(1.2, .5))
        plt.tight_layout()
        plt.savefig("{file}.png".format(file=name))
        plt.savefig("{file}.pdf".format(file=name))
        return


def graphs(kegg="kegg",go="go",ppi="ppi",group=None, top=40,term=False,FC=False,path0=None, pathinfo=None):

    if kegg == "kegg":
        keggplot().paopao()

        print(group,123)

        data = pd.read_csv("ALL_vs_DEG.kegg_Classification.xls",sep="\t")
        keggplot().contrasterm(data=data, x="Percentage", y="Classification_level2",
                               hue="type", palette=[[0.4540004, 0.78659, 0.92729262], [0.89328163, 0.54468504, 0.14280632]],
                               main="_".join(group), name="ALL_vs_DEG.KEGG_Classification")

        # RGB_to_Hex([0.4540004, 0.78659, 0.92729262]), RGB_to_Hex([0.89328163, 0.54468504, 0.14280632])

        if FC is True:
            data = pd.read_csv("Up_vs_Down.kegg_Classification.xls",sep="\t")
            keggplot().contrasterm(data=data, x="Percentage", y="Classification_level2",
                                   hue="type",
                                   main="_".join(group),name="Up_vs_Down.kegg_Classification")

    if go == "go":

        # top30
        ltop = [x for x in os.listdir() if "GO.top" in x]
        pars = [x.split(".")[2] for x in ltop]
        print(ltop,pars)
        for i,j in zip(ltop,pars):
            data = pd.read_csv(i, sep="\t")
            goplot().topgo(data=data,par=j)


        # 有向无环图
        os.chdir(path0)
        pref = "-vs-".join(group)
        #pref = "-vs-".join(group.split("_"))
        print(pref)
        dif = [x for x in os.listdir() if pref in x][0]
        os.system("Rscript ./script/topGO.r -d {dif} -m Total -j gene_go.backgroud.xls -o {outdir}/".format(dif=dif,
                                                                                                         outdir=pathinfo))
        if FC is True:
            os.system("Rscript ./script/topGO.r -d {dif} -m Up -j gene_go.backgroud.xls -o {outdir}/".format(dif=dif,outdir=pathinfo))
            os.system("Rscript ./script/topGO.r -d {dif} -m Down -j gene_go.backgroud.xls -o {outdir}/".format(dif=dif,

                                                                                                             outdir=pathinfo))
        # 对比图
        os.chdir(pathinfo)
        os.system(
            r"Rscript ../../../script/GO.level2_plot_compare.r -i ALL_vs_DEG.GO.level2.stat.xls -n {group} -m All,DEP -p ALL_vs_DEG.GO.level2.stat -o ./".format(group="-vs-".join(group)))

        if FC is True:
            os.system(
                r"Rscript ../../../script/GO.level2_plot_compare.r -i Up_vs_Down.GO.level2.stat.xls -n {group} -m Up,Down -p Up_vs_Down.GO.level2.stat -o ./".format(group="-vs-".join(group)))

        pass

    if ppi == "ppi":

        def score(term=term,FC=FC,topl=None):
            if term is True:
                data = pd.read_csv("ppi_selected_term.xls", sep="\t", index_col="term")
                term10 = data.index.tolist()

            # 根据 protein2protein_network背景 和 差异蛋白列表 准备 蛋白的连接 和 score

            dif = pd.read_csv("ppi_nodeFoldchange.xls", sep="\t", index_col=0)
            if "Gene Name" in dif.columns:
                dprogene = dif['Gene Name'].to_dict()
            else:
                dprogene = None

            edge = pd.read_csv("ppi_nodes.xls", sep="\t")
            en = len(edge[edge["combined_score"].isnull()])

            # 控制 蛋白节点 输入
            if topl is None:
                nodes_p = set(edge.iloc[:, 0].values.tolist() + edge.iloc[:,1].values.tolist())
                #nodes_p = [x for x in nodes_p if x not in term10]
            else:
                nodes_p = topl

            # 计算权重得分
            dic_score = {}
            la = edge.loc[:, "node1"].values.tolist() + edge.loc[:, "node2"].values.tolist()
            lat = Counter(la).items()
            for i,j in lat:
                dic_score[i] = j
            if term is True:
                for p in term10:
                    dic_score.pop(p)



            #
            # d1 = edge.iloc[en:, [0, 2]]
            # d2 = edge.iloc[en:, [1, 2]]
            # d2.columns = d1.columns.tolist()
            # d = pd.concat([d1, d2])
            # d = d.set_index("node1")
            # dic_score = {}
            # for gene in d.index:
            #     m = d.loc[gene, :].sum()
            #     if isinstance(m, pd.Series):
            #         m = m.values[0]
            #     dic_score[gene] = m

            dproscore = {x: dic_score[x] for x in nodes_p}

            bian = edge.iloc[:, [0, 1]].values.tolist()
            edgx = len(edge[edge["combined_score"].isnull()])
            epath = bian[0:edgx]
            # epath = [x for x in epath if x[0] in nodes_p and x[1] in nodes_p]
            epro = bian[edgx:]
            epro = [x for x in epro if x[0] in nodes_p and x[1] in nodes_p]

            if term is True:
                dpath = data["pvalue"].to_dict()
            else:
                dpath = None

            if FC is True:
                dprofc = dif["FC"][nodes_p].to_dict()
            else:
                dprofc = None



            return dproscore, epro, epath, dpath, dprofc, dprogene

        dproscore, epro, epath, dpath, dprofc, dprogene = score(term=term,FC=FC,topl=None)
        print(epath,123)
        pro_path = [x for x in set([x[1] for x in epath])]
        print(epro, 123)
        ds = sorted(dproscore.items(), key=lambda x: x[1],reverse=True)

        if top is not None:
            ds = ds[0:20]
            topl = [x[0] for x in ds] + pro_path
        else:
            topl = [x[0] for x in ds]

        dproscore, epro, epath, dpath, dprofc, dprogene = score(term=term,FC=FC,topl=topl)

        _ppi(dproscore, epro, epath=epath, dpath=dpath, dprofc=dprofc, dprogene=None,
            figname="ppi_query",layout="circular_layout", adjust=True)
        _ppi(dproscore, epro, epath=epath, dpath=dpath, dprofc=dprofc, dprogene=dprogene,
            figname="ppi_gene",layout="circular_layout", adjust=True)

    pass

#
# os.chdir(r"C:\Users\Pioneer\Desktop\Demo - 副本\enrich\GO_enrichment\Catagen_Anagen")
# data = pd.read_csv("GO.top.Total.xls", sep="\t")
# #graphs(kegg=None,go=None,ppi="ppi",top=40,term=True,FC=True)
#
# goplot().topgo(data=data)
