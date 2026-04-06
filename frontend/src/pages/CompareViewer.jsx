import React, { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, Loader2 } from 'lucide-react';

import { compareDocuments } from '../services/api';
import CompareDocumentPane from '../components/compare/CompareDocumentPane';

const normalizeTopicItems = (items, fallbackTopic) => {
  if (!Array.isArray(items)) return [];

  return items
    .map((item) => {
      if (item && typeof item === 'object') {
        const topic = String(item.topic || fallbackTopic || 'General').trim();
        const details = String(item.details || '').trim();
        if (!details) return null;
        return { topic, details };
      }

      const details = String(item || '').trim();
      if (!details) return null;
      return { topic: fallbackTopic || 'General', details };
    })
    .filter(Boolean);
};

const normalizeSummary = (summary) => {
  if (!summary || typeof summary !== 'object') {
    return {
      overview: '',
      similarities: [],
      differences: {
        docA: [],
        docB: [],
      },
    };
  }

  return {
    overview: String(summary.overview || '').trim(),
    similarities: normalizeTopicItems(summary.similarities, 'Common Ground'),
    differences: {
      docA: normalizeTopicItems(summary?.differences?.docA, 'Document A'),
      docB: normalizeTopicItems(summary?.differences?.docB, 'Document B'),
    },
  };
};

const ComparisonSection = ({ title, items, emptyMessage, palette }) => {
  const palettes = {
    emerald: {
      wrapper: 'border-emerald-200 bg-emerald-50',
      heading: 'text-emerald-900',
      card: 'border-emerald-200/80 bg-white/80',
      body: 'text-emerald-900/90',
      indicator: 'text-emerald-700',
    },
    rose: {
      wrapper: 'border-rose-200 bg-rose-50',
      heading: 'text-rose-900',
      card: 'border-rose-200/80 bg-white/80',
      body: 'text-rose-900/90',
      indicator: 'text-rose-700',
    },
    sky: {
      wrapper: 'border-sky-200 bg-sky-50',
      heading: 'text-sky-900',
      card: 'border-sky-200/80 bg-white/80',
      body: 'text-sky-900/90',
      indicator: 'text-sky-700',
    },
  };

  const styles = palettes[palette] || palettes.emerald;

  return (
    <article className={`rounded-xl border p-3 ${styles.wrapper}`}>
      <h3 className={`text-sm font-bold ${styles.heading}`}>{title}</h3>

      {items.length > 0 ? (
        <div className="mt-2 space-y-2">
          {items.map((item, index) => (
            <details key={`${title}-${index}`} className={`rounded-lg border p-2 ${styles.card}`}>
              <summary className={`flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-semibold ${styles.heading}`}>
                <span>{item.topic}</span>
                <span className={`text-xs font-medium ${styles.indicator}`}>Expand</span>
              </summary>
              <p className={`mt-2 whitespace-pre-line text-sm leading-relaxed ${styles.body}`}>{item.details}</p>
            </details>
          ))}
        </div>
      ) : (
        <p className={`mt-2 text-sm ${styles.body}`}>{emptyMessage}</p>
      )}
    </article>
  );
};

const CompareViewer = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const query = useMemo(() => (location.state?.query || '').trim(), [location.state]);
  const docIds = useMemo(() => (Array.isArray(location.state?.docIds) ? location.state.docIds : []), [location.state]);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [comparison, setComparison] = useState(null);
  const summary = normalizeSummary(comparison?.summary);

  useEffect(() => {
    if (!query || docIds.length < 2) {
      navigate('/compare', { replace: true });
      return;
    }

    let isMounted = true;

    const runCompare = async () => {
      setIsLoading(true);
      setError('');
      try {
        const payload = await compareDocuments(docIds, query);
        if (isMounted) {
          setComparison(payload);
        }
      } catch (err) {
        if (isMounted) {
          setError(err.message || 'Failed to compare documents');
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    runCompare();
    return () => {
      isMounted = false;
    };
  }, [docIds, navigate, query]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-sky-50">
      <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/95 backdrop-blur-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-4 sm:px-6">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <button
                onClick={() => navigate('/compare')}
                className="inline-flex items-center gap-1 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-semibold text-slate-700 hover:bg-slate-50"
              >
                <ArrowLeft className="h-4 w-4" />
                Back
              </button>
            </div>
            <h1 className="mt-2 truncate text-xl font-black text-slate-800">Comparison: {query}</h1>
          </div>

          <Link
            to="/chat"
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
          >
            Open Chat
          </Link>
        </div>
      </header>

      <main className="mx-auto flex max-w-7xl flex-col gap-5 px-4 py-6 sm:px-6">
        <section className="rounded-2xl border border-slate-200 bg-white shadow-sm">
          <div className="border-b border-slate-200 px-5 py-4">
            <h2 className="text-lg font-bold text-slate-800">Summary</h2>
          </div>
          <div className="px-5 py-4">
            {isLoading ? (
              <div className="flex items-center gap-2 text-sm text-slate-600">
                <Loader2 className="h-4 w-4 animate-spin" />
                Generating comparison...
              </div>
            ) : null}

            {!isLoading && error ? (
              <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>
            ) : null}

            {!isLoading && !error ? (
              <div className="space-y-3">
                <article className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="text-xs font-bold uppercase tracking-wide text-slate-600">Overview</p>
                  <p className="mt-2 whitespace-pre-line text-sm leading-relaxed text-slate-700">
                    {summary.overview || 'Detailed overview was not generated.'}
                  </p>
                </article>

                <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
                  <ComparisonSection
                    title="Similarities"
                    items={summary.similarities}
                    emptyMessage="No similarities extracted."
                    palette="emerald"
                  />
                  <ComparisonSection
                    title="Doc A Differences"
                    items={summary.differences.docA}
                    emptyMessage="No unique points extracted for Doc A."
                    palette="rose"
                  />
                  <ComparisonSection
                    title="Doc B Differences"
                    items={summary.differences.docB}
                    emptyMessage="No unique points extracted for Doc B."
                    palette="sky"
                  />
                </div>
              </div>
            ) : null}
          </div>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-bold text-slate-800">Split View</h2>
            {docIds.length > 2 ? (
              <p className="text-xs font-semibold text-slate-500">Using first two selected documents in V1</p>
            ) : null}
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <CompareDocumentPane title="Document A" documentData={comparison?.docA} tint="amber" />
            <CompareDocumentPane title="Document B" documentData={comparison?.docB} tint="sky" />
          </div>
        </section>
      </main>
    </div>
  );
};

export default CompareViewer;