# This is a sample Python script.

# Press Ctrl+P to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
# a = np.array([[1, 2, 3], [3, 4, 5]])
# b = np.array([1, 1])
# c = np.array([a[i] for i in range(len(b)) if b[i] == 1])
# print(c.mean(0))
# a = np.arctan(1)
# print(np.pi / 4)
import pyecharts.options as opts
from pyecharts.charts import Line
x=['星期一','星期二','星期三','星期四','星期五','星期七','星期日']
y1=[100,200,300,400,100,400,300]
y2=[200,300,200,100,200,300,400]
line=(
    Line()
    .add_xaxis(xaxis_data=x)
    .add_yaxis(series_name="y1线",y_axis=y1, is_smooth=True)
    .add_yaxis(series_name="y2线",y_axis=y2, is_smooth=True)
    .set_global_opts(title_opts=opts.TitleOpts(title="Line-多折线重叠"))
)
html = line.render_notebook()






