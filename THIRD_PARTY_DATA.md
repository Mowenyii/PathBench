# Third-party data and models

PathBench grants no new rights to upstream datasets, images, annotations, or
model checkpoints. Obtain each dependency from its official source and follow
its terms.

| Resource | Use in PathBench | Shipped in this repository |
|---|---|---|
| CLEVR v1.0 / CLEVR-Math (`dali-does/clevr-math`) | Synthetic scenes and questions | Object-centred crops of the 72 development and 180 confirmation scenes, one-pixel nuisance crops, manifests with ground-truth attributes and CLEVR-Math questions. CLEVR images are CC BY 4.0; the CLEVR-Math card states "Creative Commons Attribution Share Alike 4.0", so `data/clevr_*` is released under CC BY-SA 4.0 |
| RAG-IGBench (`Muyi13/RAG-IGBench`, Apache-2.0) | Natural full/crop source binding | Manifest only (source member names, crop boxes, view hashes); views are rebuilt with `scripts/prepare_ragig_views.py` |
| COCO 2017 val (images) | Natural-image pixels for the LVIS protocol | Not shipped; crops are rebuilt with `scripts/prepare_lvis_crops.py` |
| LVIS v1 val (annotations) | Instance boxes and categories | Box coordinates, instance IDs, and categories for the 120 confirmation images, inside the manifest (LVIS annotation terms apply) |
| Qwen2.5-VL-7B-Instruct, InternVL2.5-8B, LLaVA-OneVision-Qwen2-7B, Idefics2-8B | Frozen VLM inference | Not shipped; pinned revisions in `configs/pathbench.json`, downloaded with `scripts/download_models.py` |

Bundled data (`data/` manifests and crops and `results/frozen/` model outputs)
are released under CC BY-SA 4.0 because some files contain or derive from
CLEVR-Math questions. The code is released under the MIT License. Where a model
license imposes terms on generated outputs, those terms continue to apply.
