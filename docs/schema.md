# Sample Data and annotation schema

> Note: the public data package (built by `tools/make_release.py`) stores the panel arrays as
> lossless PNG (`panels/images/*.png` uint16, `panels/masks/*.png` uint8, `panels/inst/*.png` uint16)
> instead of `.npy`, adds `is_visible_expert` / `is_reject_expert` to every box, and uses
> `panels/boxes/<panel_id>.json` file names. Everything else in this schema applies unchanged.

A small, clean-background slice of MuraTrace for inspecting the data format. 
The full benchmark has 540 glasses / 30,240 panels; 
here we include one synthetic glass, a set of its annotated panels, 
the source clean backgrounds, and viewable preview renders. Panel arrays are stored as `.npy`; 
preview PNGs let you view them without loading numpy.

## Layout
```
backgrounds_clean/      Real defect-free panel backgrounds (PNG) used as I in I'=I(1±aM)
glass/
  G000000.png           Full 8x7 synthetic glass (raw image)
  route_graphs/
    G000000.json        21-node process-trace route graph for this glass
panels/                 Per-panel arrays + annotations (one glass, defective panels)
  *_img.npy             Panel luminance array, HxW (the released panel format)
  *_mask.npy            6-class semantic mask (0 bg,1 spot,2 line,3 region,4 gradient,5 band)
  *_boxes.json          Per-instance annotations (schema below)
previews/
  fig_class_examples.png  Main-paper Fig. 1 (the five visual classes / eight subtypes)
  fig_glass_overlay.png   Main-paper Fig. 2 (one 8x7 glass: raw + semantic overlay)
  semu_sweep/             Per-subtype SEMU sweeps shown to engineers in the expert study
                          (the Part-B threshold-elicitation stimuli)
```

## `*_boxes.json` schema
```
panel_id        e.g. "G000000_P00_03"
panel_size      [W, H] in pixels (727 x 359)
boxes[]         one entry per visible instance:
  instance_id, mask_type, class_id, extent, subtype, polarity
  semu, panel_instance_semu, panel_semu_peak, panel_semu_p95_*   (visibility, SEMU units)
  is_visible, is_reject                                          (thresholded labels)
  process_step, tool_id, fault_mode, time_window                 (process-trace ground truth)
  bbox [x0,y0,x1,y1], bbox_norm, yolo                            (localization)
```

## `route_graphs/*.json` schema
```
nodes[]         21 route nodes: {cycle, step, step_index, tool_id, equip_idx, node_pos}
edges[]         directed process flow between nodes
route_node_idx  lookup from (cycle, step) to node index
```
The seven steps Clean -> Coat -> Deposit -> Litho -> Etch -> Strip -> Anneal repeat
for three cycles (3x7 = 21 nodes); each defect's causal `tool_id` is bound to one
node, giving the process-trace ground truth (see supplementary.pdf S4).

> The full dataset is released with the paper.
> (all glasses, panels, masks, boxes, route graphs, the complete background bank, and generation configs) 
