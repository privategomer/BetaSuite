#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # tools/analysis/ is now 2 levels below the core dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import betaconfig
import betautils_config as bu_config
import betautils_detector as bu_detector

parts_to_blur = bu_config.get_parts_to_blur()

print("=== Resolved per-label config (via get_parts_to_blur) ===")
for label in betaconfig.items_to_censor:
    if label not in parts_to_blur:
        print("%-20s NOT PRESENT in get_parts_to_blur() output -- won't be censored at all!" % label)
        continue
    p = parts_to_blur[label]
    print("%-20s min_prob=%-6s width_area_safety=%-6s height_area_safety=%-6s time_safety=%-6s censor_style=%s" % (
        label, p.get('min_prob'), p.get('width_area_safety'), p.get('height_area_safety'), p.get('time_safety'), p.get('censor_style')
    ))

print()
print("=== Directly-read overrides (censor_shape / position_smoothing) ===")
for label in betaconfig.items_to_censor:
    override = betaconfig.item_overrides.get(label, {})
    shape = override.get('censor_shape', betaconfig.default_censor_shape)
    smoothing = override.get('position_smoothing', betaconfig.default_position_smoothing)
    print("%-20s censor_shape=%-8s position_smoothing=%s" % (label, shape, smoothing))

print()
active_backend = bu_detector.selected_backend_name()
print("=== class_suppression rules (backend: %s - per-backend as of the nudenet_v3 tuning work, see betautils_detector.get_class_suppression) ===" % (active_backend))
resolved_suppression = bu_detector.get_class_suppression( active_backend )
if not resolved_suppression:
    print("  (none - this backend has no class_suppression rules configured)")
for label, rule in resolved_suppression.items():
    print("%-20s -> %s" % (label, rule))

print()
print("=== Raw item_overrides exactly as written in betaconfig.py, for comparison ===")
for label, override in betaconfig.item_overrides.items():
    print("%-20s %s" % (label, override))
