import { useEffect, useState } from "react";

import { MarkdownDocument } from "./MarkdownDocument";

const DOCS = [
  { title: "개요", path: "/docs/overview.md", id: "overview" },
  { title: "시뮬레이션 환경", path: "/docs/simulation-environment.md", id: "simulation-environment" },
  { title: "상태와 행동", path: "/docs/mdp-formulation.md", id: "mdp-formulation" },
  { title: "제어 모델과 학습", path: "/docs/policy-and-training.md", id: "policy-and-training" },
  { title: "보상 설계", path: "/docs/reward-function.md", id: "reward-function" }
];

export function DocsPage() {
  const [contentByPath, setContentByPath] = useState<Record<string, string>>({});

  useEffect(() => {
    let cancelled = false;

    async function loadDocs() {
      const entries = await Promise.all(
        DOCS.map(async (doc) => {
          const response = await fetch(doc.path);
          const text = response.ok ? await response.text() : "문서를 불러오지 못했습니다.";
          return [doc.path, text] as const;
        })
      );

      if (!cancelled) {
        setContentByPath(Object.fromEntries(entries));
      }
    }

    void loadDocs();

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="docs-page">
      <section className="docs-page-header">
        <h1>Documentation</h1>
      </section>

      <div className="docs-layout">
        <aside className="docs-sidebar" aria-label="Documentation sections">
          <nav>
            <span>Sections</span>
            {DOCS.map((doc, index) => (
              <a href={`#${doc.id}`} key={doc.path}>
                <small>{String(index + 1).padStart(2, "0")}</small>
                <strong>{doc.title}</strong>
              </a>
            ))}
          </nav>
        </aside>

        <div className="docs-stack">
          {DOCS.map((doc) => (
            <article className="doc-section" id={doc.id} key={doc.path}>
              <MarkdownDocument content={contentByPath[doc.path] ?? "문서를 불러오는 중입니다."} />
            </article>
          ))}
        </div>
      </div>
    </main>
  );
}
