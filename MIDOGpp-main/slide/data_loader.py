import openslide
from fastai.vision import *
from fastai.data_block import *


class SlideContainer:

    def __init__(self, file: Path, y, tumortype, level: int = 0, width: int = 256, height: int = 256,
                 sample_func: callable = None):
        self.file = file
        self.slide = openslide.open_slide(str(file))
        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]
        self.y = y
        self.tumortype = tumortype
        self.sample_func = sample_func
        self.classes = list(set(self.y[1]))

        if level is None:
            level = self.slide.level_count - 1
        self.level = level

    def get_patch(self,  x: int = 0, y: int = 0):
        return np.array(self.slide.read_region(location=(int(x * self.down_factor),int(y * self.down_factor)),
                                               level=self.level, size=(self.width, self.height)))[:, :, :3]

    def get_new_train_coordinates(self):
        # use passed sampling method
        if callable(self.sample_func):
            return self.sample_func(self.y, **{"classes": self.classes, "size": self.shape,
                                               "level_dimensions": self.slide.level_dimensions,
                                               "level": self.level, "tumortype": self.tumortype})
        else:
            print("No sample function passed, returning (0,0)")
            return 0, 0

    @property
    def shape(self):
        return self.width, self.height

    def __str__(self):
        return 'SlideContainer with:\n sample func: '+str(self.sample_func)+'\n slide:'+str(self.file)


class SlideLabelList(LabelList):

    def __getitem__(self, idxs: Union[int, np.ndarray]) -> 'LabelList':
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            if self.item is None:
                slide_container = self.x.items[idxs]

                xmin, ymin = slide_container.get_new_train_coordinates()
                
                from slide.slide_helper import print_numbers
                print_numbers()

                x = self.x.get(idxs, xmin, ymin)
                y = self.y.get(idxs, xmin, ymin)
            else:
                x, y = self.item, 0
            if self.tfms or self.tfmargs:
                x = x.apply_tfms(self.tfms, **self.tfmargs)
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve': False})
            if y is None: y = 0
            return x, y
        else:
            return self.new(self.x[idxs], self.y[idxs])


PreProcessors = Union[PreProcessor, Collection[PreProcessor]]
fastai_types[PreProcessors] = 'PreProcessors'


class SlideItemList(ItemList):

    def __init__(self, items: Iterator, path: PathOrStr = '.', label_cls: Callable = None, inner_df: Any = None,
                 processor: PreProcessors = None, x: 'ItemList' = None, ignore_empty: bool = False):
        self.path = Path(path)
        self.num_parts = len(self.path.parts)
        self.items, self.x, self.ignore_empty = items, x, ignore_empty
        self.sizes = [None] * len(self.items)
        if not isinstance(self.items, np.ndarray): self.items = array(self.items, dtype=object)
        self.label_cls, self.inner_df, self.processor = ifnone(label_cls, self._label_cls), inner_df, processor
        self._label_list, self._split = SlideLabelList, ItemLists
        self.copy_new = ['x', 'label_cls', 'path']

    def __getitem__(self, idxs: int, x: int = 0, y: int = 0) -> Any:
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            return self.get(idxs, x, y)
        else:
            return self.get(*idxs)

    def label_from_list(self, labels: Iterator, label_cls: Callable = None, **kwargs) -> 'LabelList':
        "Label `self.items` with `labels`."
        labels = array(labels, dtype=object)
        label_cls = self.get_label_cls(labels, label_cls=label_cls, **kwargs)
        y = label_cls(labels, path=self.path, **kwargs)
        res = SlideLabelList(x=self, y=y)
        return res


class SlideImageItemList(SlideItemList):
    pass


class SlideObjectItemList(SlideImageItemList, ImageList):

    def get(self, i, x: int, y: int):
        fn = self.items[i]
        res = self.open(fn, x, y)
        self.sizes[i] = res.size
        return res


class ObjectItemListSlide(SlideObjectItemList):

    def open(self, fn: SlideContainer, x: int = 0, y: int = 0):
        return Image(pil2tensor(fn.get_patch(x, y) / 255., np.float32))


class SlideObjectCategoryList(ObjectCategoryList):

    def get(self, i, x: int = 0, y: int = 0):
        h, w = self.x.items[i].shape
        bboxes, labels = self.items[i]

        bboxes = np.array([box for box in bboxes]) if len(np.array(bboxes).shape) == 1 else np.array(bboxes)
        labels = np.array(labels)
        
        if len(labels) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - x
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - y

            bb_widths = (bboxes[:, 2] - bboxes[:, 0]) / 2
            bb_heights = (bboxes[:, 3] - bboxes[:, 1]) / 2

            ids = ((bboxes[:, 0] + bb_widths) > 0) \
                  & ((bboxes[:, 1] + bb_heights) > 0) \
                  & ((bboxes[:, 2] - bb_widths) < w) \
                  & ((bboxes[:, 3] - bb_heights) < h)

            bboxes = bboxes[ids]
            bboxes = np.clip(bboxes, 0, max(h, w))
            bboxes = bboxes[:, [1, 0, 3, 2]]

            labels = labels[ids]
            
            if self.x.items[i].tumortype == "human neuroendocrine tumor":
              global all_mitosis
              all_mitosis+= len(labels)
              print("NUMBER OF MITOTIC FIGURES ->"+str(all_mitosis))

        if len(labels) == 0:
            labels = np.array([0])
            bboxes = np.array([[0, 0, 1, 1]])

        return ImageBBox.create(h, w, bboxes, labels, classes=self.classes, pad_idx=self.pad_idx)