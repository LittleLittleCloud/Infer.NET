/* Progressive enhancement for the experiment notebook; without JavaScript every note and link stays visible. */
(function (scope) {
  "use strict";
  const normalize = value => String(value).toLocaleLowerCase("en-US").replace(/\s+/g, " ").trim();
  // Every whitespace-separated term must appear somewhere in the note text.
  const matches = (haystack, query) => normalize(query).split(" ").filter(Boolean)
    .every(term => haystack.includes(term));
  const countText = (shown, total, query) => normalize(query) ?
    `Showing ${shown} of ${total} notes matching “${query.trim()}”.` : `Showing all ${total} notes.`;
  const api = {normalize, matches, countText};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (!scope.document) return;
  const document = scope.document;
  function start() {
    const tools = document.getElementById("notebook-tools");
    if (!tools) return;
    const input = document.getElementById("notebook-filter"), reset = document.getElementById("notebook-reset");
    const count = document.getElementById("notebook-count"), empty = document.getElementById("notebook-empty");
    const notes = [...document.querySelectorAll("article.experiment")].map(article => ({
      article, text: normalize(article.textContent),
      entries: [...document.querySelectorAll("[data-note-link]")].filter(el => el.dataset.noteLink === article.id)
    }));
    const groups = [...document.querySelectorAll(".notebook-group")];
    function apply(announcement = "") {
      const query = input.value;
      let shown = 0;
      notes.forEach(note => {
        const visible = matches(note.text, query);
        note.article.hidden = !visible;
        note.entries.forEach(entry => { entry.hidden = !visible; });
        if (visible) shown++;
      });
      groups.forEach(group => {
        group.hidden = ![...group.querySelectorAll("[data-note-link]")].some(entry => !entry.hidden);
      });
      reset.hidden = !normalize(query);
      empty.hidden = shown > 0;
      count.textContent = (announcement ? announcement + " " : "") + countText(shown, notes.length, query);
    }
    // A deep link must always reveal its note, even if the current filter hides it.
    function revealLinked() {
      let id;
      try { id = decodeURIComponent(scope.location.hash.replace(/^#/, "")); } catch (error) { return; }
      const note = notes.find(n => n.article.id === id);
      if (!note || !note.article.hidden) return;
      input.value = "";
      apply("Filter cleared to show the linked note.");
      note.article.scrollIntoView();
    }
    tools.hidden = false;
    input.addEventListener("input", () => apply());
    reset.addEventListener("click", () => {
      input.value = "";
      apply();
      input.focus();
    });
    scope.addEventListener("hashchange", revealLinked);
    apply();
    revealLinked();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})(typeof window !== "undefined" ? window : globalThis);
