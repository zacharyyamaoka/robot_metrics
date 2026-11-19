// Colours Python tokens in every <pre class="code"><code> block: comments, strings, keywords, numbers, decorators.
// Offline and tiny on purpose: Pygments spans for 32 functions cost ~350 KB of page.
document.querySelectorAll('pre.code code').forEach(function (block) {
  var text = block.textContent, out = '', last = 0, match;
  var token = /(#[^\n]*)|("""[\s\S]*?"""|f?"(?:\\.|[^"\\\n])*"|f?'(?:\\.|[^'\\\n])*')|\b(def|return|for|in|if|else|elif|import|from|as|with|while|not|and|or|None|True|False|lambda|class|try|except|raise|yield|is|pass|break|continue)\b|\b(\d+(?:\.\d+)?(?:e-?\d+)?)\b|(@\w+)/g;
  function esc(value) { return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  while ((match = token.exec(text)) !== null) {
    out += esc(text.slice(last, match.index));
    var kind = match[1] ? 'c' : match[2] ? 's' : match[3] ? 'k' : match[4] ? 'n' : 'd';
    out += '<span class="' + kind + '">' + esc(match[0]) + '</span>';
    last = token.lastIndex;
  }
  block.innerHTML = out + esc(text.slice(last));
});

// Click any picture to see it full size; click again (or Esc) to close.
(function () {
  var box = document.getElementById('lightbox');
  if (!box) return;
  var big = box.querySelector('img');
  document.querySelectorAll('img.zoom, .card img').forEach(function (img) {
    img.addEventListener('click', function () { big.src = img.src; box.style.display = 'flex'; });
  });
  box.addEventListener('click', function () { box.style.display = 'none'; });
  document.addEventListener('keydown', function (event) { if (event.key === 'Escape') box.style.display = 'none'; });
})();
