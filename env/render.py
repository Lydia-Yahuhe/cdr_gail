import random
import copy

import cv2
import simplekml
import numpy as np

from .utils import pnpoly

# vertices_ = [
#     (109.51666666666667, 31.9),
#     (110.86666666666666, 33.53333333333333),
#     (114.07, 32.125),
#     (115.81333333333333, 32.90833333333333),
#     (115.93333333333334, 30.083333333333332),
#     (114.56666666666666, 29.033333333333335),
#     (113.12, 29.383333333333333),
#     (109.4, 29.516666666666666),
#     (109.51666666666667, 31.9),
#     (109.51666666666667, 31.9)]  # 武汉空域边界点

vertices_ = ((114.7, 32.0),
            (114.67, 31.37),
            (115.67, 30.45),
            (115.24, 30.07),
            (112.7, 30.54),
            (113.27, 31.05),
            (113.26, 31.64))
border_sector = (112.0, 116.0, 29.5, 32.5)  # 扇区实际的经纬度范围(min_x, max_y, min_y, max_y)
border_render = (112.0, 116.0, 29.5, 32.5)  # 扇区可视化的经纬度范围

# border_sector = (109.0, 116.3, 28.7, 33.8)  # 扇区实际的经纬度范围(min_x, max_y, min_y, max_y)
# border_sector = (108.6, 116.6, 28.2, 34.2)  # 扇区实际的经纬度范围(min_x, max_y, min_y, max_y)
# border_render = (109.0, 120.0, 26.0, 34.0)  # 扇区可视化的经纬度范围
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4

alt_mode = simplekml.AltitudeMode.absolute

scale_: int = 100  # BGR
channel_: int = 3
decimal: int = 1
radius_: int = 5
fps: int = 8
interval: int = 30

base_image_ = cv2.imread('./data/wuhan_base_sector.png', cv2.IMREAD_COLOR)  # 读取扇区底图（只包含扇区边界和航段线）


# ---------
# functions
# ---------
def resolution(border, scale: int):
    """
    分辨率（可视化界面的长和宽）
    假设border为[1.0, 9.0, 1.0, 7.0]，scale为100，则分辨率为800x600
    """
    min_x, max_x, min_y, max_y, *_ = border
    return (
        int((max_x - min_x) * scale),
        int((max_y - min_y) * scale)
    )


def convert_coord_to_pixel(points, border, scale: int):
    """
    将点坐标（lng, lat）转化为像素点的位置（x, y）
    """
    min_x, max_x, min_y, max_y, *_ = border
    scale_x = (max_x - min_x) * scale
    scale_y = (max_y - min_y) * scale
    return [
        (
            int((x - min_x) / (max_x - min_x) * scale_x),
            int((max_y - y) / (max_y - min_y) * scale_y)
        )
        for [x, y, *_] in points
    ]


def convert_into_image(tracks, image_path=None, limit=[], wait=None, radius=5, fx=1.0, fy=1.0, span=1):
    """
    将轨迹{t: (lng, lat, alt, spd, vspd, hdg)}转化为图像的形式
    """
    if image_path is not None:
        base_image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 读取扇区底图（只包含扇区边界和航段线）
    else:
        base_image = base_image_

    images, count = [], 0
    for t, statuses in tracks.items():
        if count % span != 0:
            count += 1
            continue

        # 底图
        image = copy.deepcopy(base_image)
        # 各个飞机的位置点
        for [a_id, lng, lat, alt, *_] in statuses:
            coord = (lng, lat)
            coord_idx = convert_coord_to_pixel([coord], border=border_sector, scale=scale_)[0]
            # 每个飞机都是个圆，圆的颜色代表飞行高度，Green（低）-Yellow-Red（高），
            range_mixed = min(510, max((alt - 6000) / 4100 * 510, 0))
            color = (0, 255, range_mixed) if range_mixed <= 255 else (0, 510 - range_mixed, 255)
            if a_id in limit:
                cv2.rectangle(
                    image,
                    (coord_idx[0] - radius, coord_idx[1] - radius),
                    (coord_idx[0] + radius, coord_idx[1] + radius),
                    color, -1
                )
            else:
                cv2.circle(image, coord_idx, radius, color, -1)

        # 图片渲染
        image = cv2.resize(image, None, fx=fx, fy=fy)

        if isinstance(wait, int):
            cv2.imshow(str(t), image)
            cv2.waitKey(wait)
            cv2.destroyAllWindows()
        images.append(image)
        count += 1
    return np.array(images)


# ---------
# opencv
# ---------
class CVRender:
    def __init__(self, video_path, image_path):
        """
        用于录制视频
        """
        self.width, self.length = resolution(border=border_render, scale=scale_)
        self.base_image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 读取扇区底图（只包含扇区边界和航段线）
        self.video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (self.width, self.length))

    def render(self, scenario, mode='human', wait=1):
        if self.video is None:
            return

        # 底图
        image = copy.deepcopy(self.base_image)
        for _ in range(10):
            frame = copy.deepcopy(image)
            # 图片渲染
            cv2.imshow(mode, frame)
            button = cv2.waitKey(wait)
            if button == 113:  # 按q键退出渲染
                return
            elif button == 112:  # 按p键加速渲染
                wait = int(wait * 0.1)
            elif button == 111:  # 按o键减速渲染
                wait *= 10
            else:
                self.video.write(frame)

    def close(self):
        if self.video is not None:
            self.video.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video = None


# ---------
# simplekml
# ---------


def make_color(red, green, blue):
    return simplekml.Color.rgb(red, green, blue, 100)


def make_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return simplekml.Color.rgb(r, g, b, 100)


def plot_line(kml, points, name='line', color=simplekml.Color.white):
    line = kml.newlinestring(
        name=name,
        coords=[tuple(p) for p in points],
        altitudemode=alt_mode,
        extrude=1
    )
    line.style.polystyle.color = color
    line.style.linestyle.color = color
    line.style.linestyle.width = 1


def plot_point(kml, point, name='point', hdg=None, description=None):
    point = kml.newpoint(
        name=name,
        coords=[point],
        altitudemode=alt_mode,
        description=description
    )
    point.style.labelstyle.scale = 0.25
    point.style.iconstyle.icon.href = 'plane.png'
    if hdg is not None:
        point.style.iconstyle.heading = (hdg + 270) % 360


def draw(tracks=None, plan=None, save_path='simplekml'):
    kml = simplekml.Kml()

    if tracks is not None:
        folder = kml.newfolder(name='real')
        for key, points in tracks.items():
            plot_line(folder, points, name=key, color=simplekml.Color.chocolate)

    if plan is not None:
        folder = kml.newfolder(name='plan')
        for key, points in plan.items():
            plot_line(folder, points, name=key, color=simplekml.Color.royalblue)

    print("Save to " + save_path + ".kml successfully!")
    kml.save(save_path + '.kml')


# ---------
# CV Demo
# ---------
def search_routing_in_sector(vertices):
    """
    将所有经过该扇区的航路筛选出来
    """
    from .load import load_data_set

    segments = {}
    check_list = []
    for key, routing in load_data_set().rou_dict.items():
        wpt_list = routing.wpt_list

        in_poly_idx = [i for i, wpt in enumerate(wpt_list) if pnpoly(vertices, wpt.location.tuple())]
        if len(in_poly_idx) <= 0:
            continue

        min_idx = max(min(in_poly_idx) - 1, 0)
        max_idx = min(len(wpt_list), max(in_poly_idx) + 2)
        new_wpt_list = wpt_list[min_idx:max_idx]
        assert len(new_wpt_list) >= 2

        for i, wpt in enumerate(new_wpt_list[1:]):
            last_wpt = new_wpt_list[i]

            name_f, name_b = last_wpt.id + '-' + wpt.id, wpt.id + '-' + last_wpt.id
            if name_f not in check_list:
                segments[name_f] = [last_wpt.location.array(), wpt.location.array()]
                check_list += [name_b, name_f]
    return segments


def generate_wuhan_base_map(vertices, border, scale, save_path=None, channel=3):
    """
    画出武汉扇区的边界线和扇区内的航段线，并保存为图片的形式
    """
    # 计算分辨率
    width, length = resolution(border, scale)
    # 创建一个的白色画布，RGB(255,255,255)为白色
    image = np.zeros((length, width, channel), np.uint8)
    # 将空域边界画在画布上
    # points = convert_coord_to_pixel(vertices, border=border, scale=scale_)
    # cv2.polylines(image,
    #               [np.array(points, np.int32).reshape((-1, 1, 2,))],
    #               isClosed=True,
    #               color=(0, 0, 0),
    #               thickness=2)
    # 将航路段画在画布上
    segments = search_routing_in_sector(vertices)
    for seg in segments.values():
        seg_idx = convert_coord_to_pixel(seg, border=border, scale=scale)
        cv2.line(
            image,
            seg_idx[0], seg_idx[1],
            (255, 255, 255), 1
        )
    # 制作成图片
    if save_path is not None:
        cv2.imwrite(save_path, image)
    # 按q结束展示
    cv2.imshow("wuhan", image)
    if cv2.waitKey(0) == 113:
        cv2.destroyAllWindows()


def main():
    # generate_wuhan_base_map(vertices=vertices_,
    #                         border=border_render,
    #                         scale=scale_,
    #                         save_path='data/wuhan_base_render.png')

    generate_wuhan_base_map(vertices=vertices_,
                            border=border_sector,
                            scale=scale_,
                            save_path='./data/wuhan_base_sector.png')


if __name__ == '__main__':
    main()
