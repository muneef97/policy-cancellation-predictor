// LightGBM tree evaluator (mirrors LightGBM's NumericalDecision / CategoricalDecision)
function predictRaw(trees, x){
  var sum = 0;
  for (var t = 0; t < trees.length; t++){
    var nodes = trees[t], i = 0;
    while (nodes[i].v === undefined){
      var n = nodes[i], v = x[n.f];
      var isNan = (v === null || v === undefined || Number.isNaN(v));
      var goLeft = undefined;
      if (n.c){                                   // categorical split
        var iv;
        if (isNan){ if (n.m === 2){ goLeft = false; } else { iv = 0; } }
        else iv = Math.trunc(v);
        if (goLeft === undefined){ goLeft = iv >= 0 && n.c.indexOf(iv) !== -1; }
      } else {                                    // numerical split
        var fv = v;
        if (isNan && n.m !== 2) fv = 0;
        if ((n.m === 1 && Math.abs(fv) <= 1e-35) || (n.m === 2 && isNan)) goLeft = n.dl === 1;
        else goLeft = fv <= n.t;
      }
      i = goLeft ? n.l : n.r;
    }
    sum += nodes[i].v;
  }
  return sum;
}
function predictProb(trees, x){ return 1 / (1 + Math.exp(-predictRaw(trees, x))); }
function calibrate(s, w){ return s / (s + w * (1 - s)); }
if (typeof module !== "undefined") module.exports = {predictRaw, predictProb, calibrate};
