from instrument_ir.data.coco_parser import parse_split


def test_excludes_umbrella_category(coco_root):
    coco = parse_split(coco_root, "valid")
    # La categoría id 0 'instruments' se excluye; quedan los 2 instrumentos reales.
    assert set(coco.categories.values()) == {"adufe", "cavaquinho"}
    assert 0 in coco.excluded_category_ids
    assert 0 not in coco.categories


def test_multilabel_image(coco_root):
    coco = parse_split(coco_root, "valid")
    by_id = {im.coco_id: im for im in coco.images}
    # img 0 tiene adufe y cavaquinho (multi-etiqueta).
    assert by_id[0].category_names == ("adufe", "cavaquinho")
    assert by_id[1].category_names == ("cavaquinho",)
    assert by_id[2].category_names == ("adufe",)


def test_split_alias_detection(coco_root):
    # parse_split debe encontrar 'valid' (alias de validación).
    coco = parse_split(coco_root, "valid")
    assert len(coco.images) == 3
