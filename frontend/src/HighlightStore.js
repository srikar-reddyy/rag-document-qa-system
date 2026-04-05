/**
 * Highlight Store (lightweight local state helpers)
 */

export const createInitialHighlightState = () => ({
  activeDoc: null,
  activePage: 1,
  activeBoxes: [],
  activeBbox: null,
  trigger: 0,
});

const normalizeBoxes = (boxes) => {
  if (!Array.isArray(boxes)) return [];

  return boxes
    .map((box) => (Array.isArray(box) ? box.map((v) => Number(v)) : null))
    .filter((box) => Array.isArray(box) && box.length >= 4 && box.every((v) => Number.isFinite(v)));
};

const normalizeBbox = (bbox) => {
  if (!Array.isArray(bbox) || bbox.length < 4) return null;

  const parsed = bbox.slice(0, 4).map((value) => Number(value));
  if (!parsed.every((value) => Number.isFinite(value))) return null;
  return parsed;
};

export const buildHighlightState = (highlight) => {
  const normalizedBoxes = normalizeBoxes(highlight?.boxes);
  const legacyBbox = normalizeBbox(highlight?.bbox);
  const legacyBox = legacyBbox
    ? [
        legacyBbox[0],
        legacyBbox[1],
        legacyBbox[0] + legacyBbox[2],
        legacyBbox[1] + legacyBbox[3],
      ]
    : null;

  const activeBoxes = normalizedBoxes.length ? normalizedBoxes : (legacyBox ? [legacyBox] : []);
  const firstBox = activeBoxes[0];

  return {
    activeDoc: highlight?.doc_name || null,
    activePage: Number(highlight?.page) || 1,
    activeBoxes,
    activeBbox: firstBox
      ? [
          firstBox[0],
          firstBox[1],
          Math.max(0, firstBox[2] - firstBox[0]),
          Math.max(0, firstBox[3] - firstBox[1]),
        ]
      : legacyBbox,
    trigger: Date.now(),
  };
};
