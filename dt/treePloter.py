import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_node(ax, text, center_pos, parent_pos, leaf_node=True):
    """
    xy(parent_pos) -> xytext(center_pos)
    """
    decision_node_style = {"arrowstyle": "<|-"}

    bbox_props = {
        "boxstyle": "round,pad=0.3",
        "fc": ("green" if leaf_node else "brown"),
        "ec": "k",
        "lw": 2
    }

    kw = {
        "s": text,
        "xy": parent_pos,
        "xytext": center_pos,
        "arrowprops": decision_node_style,
        "bbox": bbox_props,
        "ha": "center",
        "va": "center",
    }
    ax.annotate(**kw)


def plot_mid_text(ax, text, center_pos, parent_pos):
    x = (center_pos[0] - parent_pos[0]) / 2 + parent_pos[0]
    y = (center_pos[1] - parent_pos[1]) / 2 + parent_pos[1]
    ax.text(x, y, text)


def plot_tree(ax, tree, text, parent_pos, p):

    leaf_num = tree.leaf_num
    center_pos = (p["x_off"] + (1 + leaf_num) /
                  (2 * p["total_leaf_num"]), p["y_off"])

    plot_node(ax, tree.feature_index, center_pos, parent_pos, False)
    plot_mid_text(ax, text, center_pos, parent_pos)

    p["y_off"] -= 1 / p["total_hight"]
    for key in tree.subtree.keys():
        if tree.subtree[key].is_leaf:
            p["x_off"] += 1 / p["total_leaf_num"]
            x_off = p["x_off"]
            y_off = p["y_off"]
            plot_node(ax, str(tree.subtree[key].leaf_class),
                      (x_off, y_off), center_pos, True)
            plot_mid_text(ax, key, (x_off, y_off), center_pos)
        else:
            plot_tree(ax, tree.subtree[key], key, center_pos, p)
    p["y_off"] += 1 / p["total_hight"]


def plot_decisionTree(tree, display=False):
    p = {
        "x_off": - 1.0 / (2 * tree.leaf_num),
        "y_off": 1.0,
        "total_leaf_num": tree.leaf_num,
        "total_hight": tree.hight
    }
    fig, ax = plt.subplots()
    ax.set_axis_off()
    plot_tree(ax, tree, "", (0.5, 1.0), p)
    if display:
        fig.show()
