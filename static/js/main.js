/* FinSight interactions: navigation, reveal-on-scroll, number counters, page transitions, forms, tabs, confusion matrix. */
(function () {
  "use strict";
  const root = document.documentElement;
  root.classList.add("js");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, c) => (c || document).querySelector(s);
  const $$ = (s, c) => Array.from((c || document).querySelectorAll(s));

  // ---- mobile navigation
  const toggle = $(".nav__toggle"), links = $(".nav__links");
  if (toggle && links) {
    toggle.addEventListener("click", () => {
      const open = links.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
    });
  }

  // ---- reveal on scroll (also drives CSS animations of bars/meters)
  const revealables = $$(".reveal, [data-animate]");
  if (reduce || !("IntersectionObserver" in window)) {
    revealables.forEach((el) => el.classList.add("is-visible"));
  } else {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("is-visible"); io.unobserve(e.target); } });
    }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
    revealables.forEach((el) => io.observe(el));
  }

  // ---- number counters: <span data-count="0.739" data-decimals="3" data-suffix="%">
  function format(v, d) { return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }); }
  function runCounter(el) {
    const target = parseFloat(el.dataset.count);
    if (isNaN(target)) return;
    const d = parseInt(el.dataset.decimals || "0", 10), suffix = el.dataset.suffix || "", prefix = el.dataset.prefix || "";
    if (reduce) { el.textContent = prefix + format(target, d) + suffix; return; }
    const dur = 1100, t0 = performance.now();
    (function tick(now) {
      const p = Math.min(1, (now - t0) / dur), eased = 1 - Math.pow(1 - p, 4);
      el.textContent = prefix + format(target * eased, d) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    })(t0);
  }
  const counters = $$("[data-count]");
  if (counters.length) {
    if (!("IntersectionObserver" in window)) counters.forEach(runCounter);
    else {
      const cio = new IntersectionObserver((entries) => {
        entries.forEach((e) => { if (e.isIntersecting) { runCounter(e.target); cio.unobserve(e.target); } });
      }, { threshold: 0.4 });
      counters.forEach((c) => cio.observe(c));
    }
  }

  // ---- page transitions on internal navigation
  if (!reduce) {
    document.addEventListener("click", (e) => {
      const a = e.target.closest("a[href]");
      if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      if (a.target === "_blank" || a.hasAttribute("download") || a.origin !== location.origin) return;
      if (a.pathname === location.pathname && a.search === location.search) return;
      if (a.pathname.startsWith("/download/") || a.pathname.endsWith("/export")) return;
      e.preventDefault();
      document.body.classList.add("is-leaving");
      setTimeout(() => { location.href = a.href; }, 170);
    });
    window.addEventListener("pageshow", (e) => { if (e.persisted) document.body.classList.remove("is-leaving"); });
  }

  // ---- buttons: loading state on submit
  $$("form[data-loading]").forEach((form) => {
    form.addEventListener("submit", () => {
      const btn = $("button[type=submit], button:not([type])", form);
      if (btn && form.checkValidity()) { btn.classList.add("is-loading"); btn.dataset.label = btn.textContent; btn.firstChild.textContent = form.dataset.loading; }
    });
  });

  // ---- tabs
  $$("[data-tabs]").forEach((wrap) => {
    const tabs = $$("[role=tab]", wrap), panels = tabs.map((t) => document.getElementById(t.getAttribute("aria-controls")));
    function select(i) {
      tabs.forEach((t, j) => { t.setAttribute("aria-selected", String(i === j)); t.tabIndex = i === j ? 0 : -1; });
      panels.forEach((p, j) => { p.hidden = i !== j; });
    }
    tabs.forEach((t, i) => {
      t.addEventListener("click", () => select(i));
      t.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const n = (i + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length; select(n); tabs[n].focus(); }
      });
    });
    select(Math.max(0, tabs.findIndex((t) => t.dataset.initial === "true")));
  });

  // ---- text area: example chips + character counter
  const ta = $("#text");
  if (ta) {
    const counter = $("#text-count"), max = parseInt(ta.getAttribute("maxlength") || "2000", 10);
    const update = () => { if (counter) counter.textContent = ta.value.length + " / " + max; };
    ta.addEventListener("input", update); update();
    $$("[data-example]").forEach((b) => b.addEventListener("click", () => { ta.value = b.dataset.example; update(); ta.focus(); }));
  }

  // ---- bring a fresh result into view after a form submit
  const result = $("#result");
  if (result) setTimeout(() => result.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" }), 120);

  // ---- CSV dropzone with client-side pre-checks (the server validates again)
  const dz = $(".dropzone");
  if (dz) {
    const input = $("input[type=file]", dz), name = $("#file-name"), err = $("#file-error"), MAX = parseInt(dz.dataset.maxBytes || "2097152", 10);
    function fail(msg) { err.textContent = msg; err.hidden = false; input.value = ""; name.textContent = "Drop a CSV file here, or click to browse"; }
    function show(file) {
      err.hidden = true; err.textContent = "";
      if (!file) { name.textContent = "Drop a CSV file here, or click to browse"; return; }
      if (!/\.csv$/i.test(file.name)) return fail("Only .csv files are accepted.");
      if (file.size > MAX) return fail("That file is larger than " + (MAX / 1048576) + " MB.");
      name.textContent = file.name + " (" + (file.size / 1024).toFixed(1) + " KB)";
    }
    input.addEventListener("change", () => show(input.files[0]));
    ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("is-over"); }));
    dz.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; show(input.files[0]); } });
  }

  // ---- confusion matrix explorer (data comes from the backend as JSON)
  const cmData = $("#cm-data"), cmGrid = $("#cm-grid");
  if (cmData && cmGrid) {
    const data = JSON.parse(cmData.textContent), labels = ["negative", "neutral", "positive"];
    const title = $("#cm-title"), perClass = $("#cm-perclass tbody"), summary = $("#cm-summary");
    function render(key) {
      const d = data[key], m = d.matrix;
      cmGrid.innerHTML = '<div class="cm__h"></div>' + labels.map((l) => '<div class="cm__h">pred<br>' + l + "</div>").join("");
      m.forEach((row, i) => {
        const total = row.reduce((a, b) => a + b, 0) || 1;
        cmGrid.insertAdjacentHTML("beforeend", '<div class="cm__h">true<br>' + labels[i] + "</div>");
        row.forEach((v, j) => {
          const share = v / total, strong = i === j && share > 0.45, wrong = i !== j;
          cmGrid.insertAdjacentHTML("beforeend", '<div class="cm__c' + (strong ? " is-strong" : "") + (wrong ? " is-wrong" : "") + '" style="--i:' + share.toFixed(3) +
            '" title="' + v + " of " + total + ' true ' + labels[i] + ' predicted ' + labels[j] + '"><b>' + v + "</b><span>" + Math.round(share * 100) + "%</span></div>");
        });
      });
      title.textContent = d.title;
      summary.textContent = "Accuracy " + d.accuracy.toFixed(3) + " | macro-F1 " + d.macro_f1.toFixed(3);
      perClass.innerHTML = labels.map((l) => {
        const c = d.per_class[l];
        return "<tr><td>" + l + '</td><td class="num">' + c.precision.toFixed(3) + '</td><td class="num">' + c.recall.toFixed(3) + '</td><td class="num">' + c.f1.toFixed(3) + '</td><td class="num">' + c.support + "</td></tr>";
      }).join("");
      $$("[data-cm]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.cm === key)));
    }
    $$("[data-cm]").forEach((b) => b.addEventListener("click", () => render(b.dataset.cm)));
    render(Object.keys(data).includes("Production") ? "Production" : Object.keys(data)[0]);
  }
})();
