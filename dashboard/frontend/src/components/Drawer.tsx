import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import s from "../App.module.css";
export default function Drawer({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    dialog.current?.showModal();
    return () => dialog.current?.close();
  }, []);
  return (
    <dialog
      ref={dialog}
      className={s.drawer}
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      aria-label={title}
    >
      <div className={s.drawerHeader}>
        <h2>{title}</h2>
        <button aria-label="关闭侧栏" onClick={onClose}>
          ✕
        </button>
      </div>
      <div className={s.drawerBody}>{children}</div>
    </dialog>
  );
}
