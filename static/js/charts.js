/* FinSight charts: declarative Chart.js wrappers driven by data-* attributes on <canvas>.
   <canvas data-chart="bar|hbar|doughnut" data-labels='[...]' data-series='[{"label":"..","data":[..],"color":"copper"}]' data-options='{...}'>
   All numbers come from the backend (rendered into the attributes by Jinja). Charts animate when scrolled into view. */
(function () {
  "use strict";
  if (!window.Chart) { return; }
  const css = getComputedStyle(document.documentElement);
  const token = (n) => css.getPropertyValue("--" + n).trim() || n;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  Chart.defaults.font.family = css.getPropertyValue("--font-sans").trim();
  Chart.defaults.font.size = 12;
  Chart.defaults.animation = reduce ? false : { duration: 1000, easing: "easeOutQuart" };
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.tooltip.backgroundColor = token("charcoal");
  Chart.defaults.plugins.tooltip.titleColor = token("paper");
  Chart.defaults.plugins.tooltip.bodyColor = token("paper");
  Chart.defaults.plugins.tooltip.cornerRadius = 2;
  Chart.defaults.plugins.tooltip.padding = 10;

  // JSON.parse(null) returns null (it does not throw), so a missing attribute must fall back explicitly.
  const json = (el, a, fallback) => { try { const v = JSON.parse(el.getAttribute(a)); return v == null ? fallback : v; } catch (e) { return fallback; } };

  function build(cv) {
    const type = cv.dataset.chart, labels = json(cv, "data-labels", []), series = json(cv, "data-series", []), o = json(cv, "data-options", {});
    const dark = !!cv.closest(".section--dark");
    const text = dark ? token("paper") : token("charcoal"), gridc = dark ? "rgba(242,239,230,.14)" : "rgba(37,39,36,.12)";
    const suffix = o.suffix || "", dec = o.decimals == null ? 0 : o.decimals;
    const fmt = (v) => Number(v).toFixed(dec) + suffix;
    // On charcoal backgrounds the dark brand colours would vanish, so use lighter equivalents (same meaning).
    const onDark = { burgundy: "#B96B68", negative: "#B96B68", forest: token("sage"), positive: token("sage") };
    const pick = (n) => (dark && onDark[n]) || token(n);
    const datasets = series.map((s) => {
      const colors = s.colors ? s.colors.map(pick) : pick(s.color || "copper");
      return { label: s.label, data: s.data, backgroundColor: colors, borderColor: type === "doughnut" ? token(dark ? "charcoal" : "paper") : colors,
               borderWidth: type === "doughnut" ? 3 : 0, borderRadius: type === "doughnut" ? 0 : 1, maxBarThickness: 34 };
    });
    const common = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: o.legend !== false && (series.length > 1 || type === "doughnut"), labels: { color: text }, position: "bottom" },
      tooltip: { callbacks: { label: (c) => " " + (c.dataset.label ? c.dataset.label + ": " : "") + fmt(c.parsed.y != null && type !== "hbar" && type !== "doughnut" ? c.parsed.y : (type === "doughnut" ? c.parsed : c.parsed.x)) } } } };
    let cfg;
    if (type === "doughnut") {
      cfg = { type: "doughnut", data: { labels, datasets }, options: { ...common, cutout: "62%", plugins: { ...common.plugins, legend: { ...common.plugins.legend, display: true } } } };
    } else {
      const horizontal = type === "hbar", stacked = !!o.stacked;
      const valueAxis = { min: o.min == null ? 0 : o.min, max: o.max, stacked, grid: { color: gridc }, border: { display: false }, ticks: { color: text, callback: (v) => Number(v).toFixed(dec) + suffix } };
      const catAxis = { stacked, grid: { display: false }, ticks: { color: text, autoSkip: false, maxRotation: 0 }, border: { color: text } };
      cfg = { type: "bar", data: { labels, datasets }, options: { ...common, indexAxis: horizontal ? "y" : "x", scales: horizontal ? { x: valueAxis, y: catAxis } : { x: catAxis, y: valueAxis } } };
    }
    new Chart(cv, cfg);
  }

  const targets = Array.from(document.querySelectorAll("canvas[data-chart]"));
  if (!("IntersectionObserver" in window)) { targets.forEach(build); return; }
  const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { build(e.target); io.unobserve(e.target); } }), { threshold: 0.2 });
  targets.forEach((c) => io.observe(c));
})();
