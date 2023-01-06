window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    },
    loader: {load: ['[tex]/ams']},
    tex: {packages: {'[+]': ['ams']}}
  };
  
  document$.subscribe(() => {
    MathJax.typesetPromise()
  })
