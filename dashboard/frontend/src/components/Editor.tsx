import { useEffect, useRef } from "react";
import { EditorView, keymap } from "@codemirror/view";
import { EditorState } from "@codemirror/state";
import { basicSetup } from "codemirror";
import { python } from "@codemirror/lang-python";
import { setDiagnostics } from "@codemirror/lint";
import type { Diagnostic } from "../api/types";
export default function Editor({
  value,
  onChange,
  diagnostics,
  onRun,
  readOnly = false,
}: {
  value: string;
  onChange?: (value: string) => void;
  diagnostics: Diagnostic[];
  onRun: () => void;
  readOnly?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null),
    view = useRef<EditorView | null>(null),
    change = useRef<(value: string) => void>(() => undefined),
    run = useRef(onRun);
  change.current = onChange || (() => undefined);
  run.current = onRun;
  useEffect(() => {
    if (!host.current) return;
    const editor = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: value,
        extensions: [
          basicSetup,
          python(),
          EditorView.contentAttributes.of({
            "aria-label": "程序编辑器",
            spellcheck: "false",
          }),
          EditorState.readOnly.of(readOnly),
          EditorView.editable.of(!readOnly),
          keymap.of(
            readOnly
              ? []
              : [
                  {
                    key: "Mod-Enter",
                    run: () => {
                      run.current();
                      return true;
                    },
                  },
                ],
          ),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) change.current(update.state.doc.toString());
          }),
          EditorView.theme({
            "&": { height: "100%", minHeight: "0", fontSize: "13px" },
            ".cm-scroller": {
              fontFamily: '"IBM Plex Mono", monospace',
              minHeight: "0",
              overflowX: "auto",
              overflowY: "scroll",
              scrollbarGutter: "stable",
              scrollbarColor: "#aebbd0 #edf2f8",
              overscrollBehavior: "contain",
              lineHeight: "1.9",
            },
            ".cm-scroller::-webkit-scrollbar": { width: "12px", height: "12px" },
            ".cm-scroller::-webkit-scrollbar-track": { background: "#edf2f8" },
            ".cm-scroller::-webkit-scrollbar-thumb": {
              background: "#aebbd0",
              border: "2px solid #edf2f8",
              borderRadius: "6px",
            },
            ".cm-content": { padding: "20px 0" },
            ".cm-gutters": {
              background: "#FAFBFD",
              border: "none",
              color: "#A0ABBB",
            },
            ".cm-lineNumbers .cm-gutterElement": { padding: "0 16px 0 12px" },
            ".cm-activeLine": { background: "#F1F5FC" },
            ".cm-activeLineGutter": { background: "#EDF2FA" },
            ".cm-cursor": { borderLeftColor: "#2458D3" },
            "&.cm-focused": { outline: "none" },
            ".cm-selectionBackground": { background: "#D9E6FF!important" },
          }),
        ],
      }),
    });
    view.current = editor;
    return () => {
      editor.destroy();
      view.current = null;
    };
  }, []);
  useEffect(() => {
    const editor = view.current;
    if (editor && editor.state.doc.toString() !== value)
      editor.dispatch({
        changes: { from: 0, to: editor.state.doc.length, insert: value },
      });
  }, [value]);
  useEffect(() => {
    const editor = view.current;
    if (!editor) return;
    editor.dispatch(
      setDiagnostics(
        editor.state,
        diagnostics.map((d) => {
          const line = editor.state.doc.line(
            Math.max(1, Math.min(editor.state.doc.lines, d.line || 1)),
          );
          return {
            from: line.from,
            to: line.to,
            severity: "error",
            message: d.message,
          };
        }),
      ),
    );
    if (diagnostics[0]?.line) {
      const line = editor.state.doc.line(
        Math.min(editor.state.doc.lines, Math.max(1, diagnostics[0].line)),
      );
      editor.dispatch({
        effects: EditorView.scrollIntoView(line.from, { y: "center" }),
      });
    }
  }, [diagnostics]);
  return <div ref={host} style={{ height: "100%", minHeight: 0 }} />;
}
