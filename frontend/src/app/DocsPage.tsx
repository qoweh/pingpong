import { useEffect, useState } from "react";

const DOCS = [
  { title: "Overview", path: "/docs/overview.md", id: "overview" },
  { title: "Simulation Environment", path: "/docs/simulation-environment.md", id: "simulation-environment" },
  { title: "MDP Formulation", path: "/docs/mdp-formulation.md", id: "mdp-formulation" },
  { title: "Policy and Training", path: "/docs/policy-and-training.md", id: "policy-and-training" },
  { title: "Web Deployment", path: "/docs/web-deployment.md", id: "web-deployment" },
  { title: "Results", path: "/docs/results.md", id: "results" },
  { title: "Problem Report", path: "/docs/problem-resolution-report.md", id: "problem-report" }
];

export function DocsPage() {
  const [contentByPath, setContentByPath] = useState<Record<string, string>>({});

  useEffect(() => {
    let cancelled = false;

    async function loadDocs() {
      const entries = await Promise.all(
        DOCS.map(async (doc) => {
          const response = await fetch(doc.path);
          const text = response.ok ? await response.text() : `Failed to load ${doc.path}`;
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
        <h1>Docs</h1>
      </section>

      <nav className="docs-grid docs-page-grid" aria-label="Documentation sections">
        {DOCS.map((doc) => (
          <a href={`#${doc.id}`} key={doc.path}>
            {doc.title}
          </a>
        ))}
      </nav>

      <div className="docs-stack">
        {DOCS.map((doc) => (
          <article className="doc-section" id={doc.id} key={doc.path}>
            <h2>{doc.title}</h2>
            <pre>{contentByPath[doc.path] ?? "Loading..."}</pre>
          </article>
        ))}
      </div>
    </main>
  );
}
