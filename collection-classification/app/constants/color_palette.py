#!/usr/bin/env python35
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageDraw
from enum import Enum
from typing import List
import logging

LOGGER = logging.getLogger(__name__)


class Label():

    def __init__(self, color_palette: List[List[int]], labels: List[str]):
        if len(color_palette) != len(labels):
            raise 'palette and labels need same number of elements. pallete:{}, labels:{}'.format(
                len(color_palette), len(labels))

        self.color_palette = color_palette
        self.labels = labels
        self.label_length = len(self.color_palette)

        self.legend_size = 12
        self.legend_margin = 5

        self.legend = self.create_legend()

    def get_colors(self):
        return self.color_palette

    def create_legend(self) -> Image:
        height = self.legend_margin + \
            (len(self.color_palette) * (self.legend_size + self.legend_margin))

        width = 100
        legend = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(legend)
        for i, (color, label) in enumerate(zip(self.color_palette, self.labels)):
            x0 = self.legend_margin
            y0 = (i * (self.legend_margin + self.legend_size)) + \
                self.legend_margin

            draw.rectangle(
                [(x0, y0), (x0 + self.legend_size, y0 + self.legend_size)], fill=tuple(color))
            draw.text([(x0 + self.legend_size + 2, y0)], label, fill=(0, 0, 0))

        return legend

    def to_image(self, seg_label: np.ndarray) -> Image:
        label_colors = self.color_palette
        shape = seg_label.shape
        rgb_image = np.zeros((shape[0], shape[1], 3))
        for h, tmp in enumerate(seg_label):
            for w, label in enumerate(tmp):
                rgb_image[h, w] = label_colors[int(label)]

        img = Image.fromarray(np.uint8(rgb_image))
        img.paste(self.legend, (10, 5))

        return img


class Label56Types(Label):

    def __init__(self):
        null = np.asarray([0, 0, 0])  # 0
        tights = np.asarray([153, 51, 0])  # 1
        shorts = np.asarray([51, 51, 0])  # 2
        blazer = np.asarray([0, 51, 0])  # 3
        t_shirt = np.asarray([0, 51, 102])  # 4
        bag = np.asarray([0, 0, 128])  # 5
        shoes = np.asarray([51, 51, 153])  # 6
        coat = np.asarray([51, 51, 51])  # 7
        skirt = np.asarray([128, 0, 0])  # 8
        purse = np.asarray([255, 102, 0])  # 9
        boots = np.asarray([128, 128, 0])  # 10
        blouse = np.asarray([0, 128, 0])  # 11
        jacket = np.asarray([0, 128, 128])  # 12
        bra = np.asarray([0, 0, 255])  # 13
        dress = np.asarray([102, 102, 153])  # 14
        pants = np.asarray([128, 128, 128])  # 15
        sweater = np.asarray([255, 0, 0])  # 16
        shirt = np.asarray([255, 153, 0])  # 17
        jeans = np.asarray([153, 204, 0])  # 18
        leggings = np.asarray([51, 153, 102])  # 19
        scarf = np.asarray([51, 204, 204])  # 20
        hat = np.asarray([51, 102, 255])  # 21
        top = np.asarray([128, 0, 128])  # 22
        cardigan = np.asarray([150, 150, 150])  # 23
        accessories = np.asarray([255, 0, 255])  # 24
        vest = np.asarray([255, 204, 0])  # 25
        sunglasses = np.asarray([255, 255, 0])  # 26
        belt = np.asarray([0, 255, 0])  # 27
        socks = np.asarray([0, 255, 255])  # 28
        glasses = np.asarray([0, 204, 255])  # 29
        intimate = np.asarray([153, 51, 102])  # 30
        stockings = np.asarray([192, 192, 192])  # 31
        necklace = np.asarray([255, 153, 204])  # 32
        cape = np.asarray([255, 204, 153])  # 33
        jumper = np.asarray([255, 255, 153])  # 34
        sweatshirt = np.asarray([204, 255, 204])  # 35
        suit = np.asarray([204, 255, 255])  # 36
        bracelet = np.asarray([153, 204, 255])  # 37
        heels = np.asarray([204, 153, 255])  # 38
        wedges = np.asarray([255, 255, 255])  # 39
        ring = np.asarray([153, 153, 255])  # 40
        flats = np.asarray([153, 51, 102])  # 41
        tie = np.asarray([255, 255, 204])  # 42
        romper = np.asarray([204, 255, 255])  # 43
        sandals = np.asarray([102, 0, 102])  # 44
        earrings = np.asarray([255, 128, 128])  # 45
        gloves = np.asarray([0, 102, 204])  # 46
        sneakers = np.asarray([204, 204, 255])  # 47
        clogs = np.asarray([0, 0, 128])  # 48
        watch = np.asarray([255, 0, 255])  # 49
        pumps = np.asarray([255, 255, 0])  # 50
        wallet = np.asarray([0, 255, 255])  # 51
        bodysuit = np.asarray([128, 0, 128])  # 52
        loafers = np.asarray([128, 0, 0])  # 53
        hair = np.asarray([0, 128, 128])  # 54
        skin = np.asarray([0, 0, 255])  # 55

        label_colors = np.array([null, tights, shorts, blazer, t_shirt, bag,
                                 shoes, coat, skirt, purse, boots, blouse, jacket,
                                 bra, dress, pants, sweater, shirt, jeans, leggings,
                                 scarf, hat, top, cardigan, accessories, vest, sunglasses,
                                 belt, socks, glasses, intimate, stockings, necklace,
                                 cape, jumper, sweatshirt, suit, bracelet, heels,
                                 wedges, ring, flats, tie, romper, sandals, earrings,
                                 gloves, sneakers, clogs, watch, pumps, wallet, bodysuit,
                                 loafers, hair, skin])

        super().__init__(label_colors, [str(i) for i in range(56)])


class Label25Types(Label):

    def __init__(self):
        background = np.asarray([0, 0, 0])  # 0
        skin = np.asarray([153, 51, 0])  # 1
        hair = np.asarray([51, 51, 0])  # 2
        bag = np.asarray([0, 51, 0])  # 3
        belt = np.asarray([0, 51, 102])  # 4
        boots = np.asarray([0, 0, 128])  # 5
        coat = np.asarray([51, 51, 153])  # 6
        dress = np.asarray([51, 51, 51])  # 7
        glasses = np.asarray([128, 0, 0])  # 8
        gloves = np.asarray([255, 102, 0])  # 9
        hat_headband = np.asarray([128, 128, 0])  # 10
        jacket_blazer = np.asarray([0, 128, 0])  # 11
        necklace = np.asarray([0, 128, 128])  # 12
        pants_jeans = np.asarray([0, 0, 255])  # 13
        scarf_tie = np.asarray([102, 102, 153])  # 14
        shirt_blouse = np.asarray([128, 128, 128])  # 15
        shoes = np.asarray([255, 0, 0])  # 16
        shorts = np.asarray([255, 153, 0])  # 17
        skirt = np.asarray([153, 204, 0])  # 18
        socks = np.asarray([51, 153, 102])  # 19
        sweater_cardigan = np.asarray([51, 204, 204])  # 20
        tights_leggings = np.asarray([51, 102, 255])  # 21
        top_tshirt = np.asarray([128, 0, 128])  # 22
        vest = np.asarray([150, 150, 150])  # 23
        watch_bracelet = np.asarray([255, 0, 255])  # 24

        label_colors = np.array([background, skin, hair, bag, belt,
                                 boots, coat, dress, glasses, gloves,
                                 hat_headband, jacket_blazer, necklace, pants_jeans, scarf_tie,
                                 shirt_blouse, shoes, shorts, skirt, socks,
                                 sweater_cardigan, tights_leggings, top_tshirt, vest, watch_bracelet])

        super().__init__(label_colors, [str(i) for i in range(25)])


class Label16Types(Label):

    def __init__(self):
        null = np.asarray([0, 0, 0])  # 0
        tights = np.asarray([153, 51, 0])  # 1
        shorts = np.asarray([51, 51, 0])  # 2
        blazer = np.asarray([0, 51, 0])  # 3
        t_shirt = np.asarray([0, 51, 102])  # 4
        bag = np.asarray([0, 0, 128])  # 5
        shoes = np.asarray([51, 51, 153])  # 6
        coat = np.asarray([51, 51, 51])  # 7
        skirt = np.asarray([128, 0, 0])  # 8
        purse = np.asarray([255, 102, 0])  # 9
        boots = np.asarray([128, 128, 0])  # 10
        blouse = np.asarray([0, 128, 0])  # 11
        jacket = np.asarray([0, 128, 128])  # 12
        bra = np.asarray([0, 0, 255])  # 13
        dress = np.asarray([102, 102, 153])  # 14
        pants = np.asarray([128, 128, 128])  # 15

        label_colors = np.array([null, tights, shorts, blazer, t_shirt, bag,
                                 shoes, coat, skirt, purse, boots, blouse, jacket,
                                 bra, dress, pants])

        super().__init__(label_colors, ["background",
                                        "all in one",
                                        "cut and sewn",
                                        "cardigan",
                                        "coat",
                                        "shirt",
                                        "jacket",
                                        "skirt",
                                        "knit",
                                        "pants",
                                        "hoodie",
                                        "bustier",
                                        "blouse",
                                        "blouson",
                                        "pullover",
                                        "dress"])


class Label5Types(Label):

    def __init__(self):
        background = np.asarray([0, 0, 0])  # 0
        tops = np.asarray([211, 222, 241])  # 1
        bottoms = np.asarray([250, 128, 114])  # 2
        accessory = np.asarray([144, 238, 144])  # 3
        person = np.asarray([153, 51, 0])  # 4

        label_colors = np.array([background, tops, bottoms, accessory, person])

        super().__init__(label_colors, [str(i) for i in range(5)])

    def reduce_from_25type(self, label_image: np.ndarray) -> np.ndarray:
        integrating_list = [(0,),  # backdround
                            (6, 7, 11, 15, 20, 22, 23),  # tops
                            (4, 13, 17, 18, 21),  # bottoms
                            (3, 5, 8, 9, 10, 12, 14, 16, 19, 24),  # accessory
                            (1, 2)]  # person

        ret_label = np.zeros(label_image.shape, dtype=np.int)
        for i, labels in enumerate(integrating_list):
            for label in labels:
                ret_label[label_image == label] = i

        return ret_label


class Label3Types(Label):

    def __init__(self):
        background = np.asarray([0, 0, 0])  # 0
        tops = np.asarray([211, 222, 241])  # 1
        bottoms = np.asarray([250, 128, 114])  # 2

        label_colors = np.array([background, tops, bottoms])

        super().__init__(label_colors, [str(i) for i in range(3)])

    def reduce_from_16type(self, label_image: np.ndarray) -> np.ndarray:
        integrating_list = [(0,),  # backdround
                            (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15),  # tops
                            (7, 9)]  # bottoms

        ret_label = np.zeros(label_image.shape, dtype=np.int)
        for i, labels in enumerate(integrating_list):
            for label in labels:
                ret_label[label_image == label] = i

        return ret_label


class LabelEnum(Enum):

    label56 = Label56Types()  # type: Label
    label25 = Label25Types()  # type: Label
    label16 = Label16Types()  # type: Label
    label5 = Label5Types()  # type: Label
    label3 = Label3Types()  # type: Label

    @classmethod
    def of(self, label_length: int) -> 'Label':
        for e in LabelEnum.__members__.values():
            if e.value.label_length == label_length:
                return e.value

        raise
