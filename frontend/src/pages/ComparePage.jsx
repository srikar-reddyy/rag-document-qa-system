import React, { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowRightLeft, FileText, Loader2 } from 'lucide-react';

import { fetchDocuments } from '../services/api';

const formatFileSize = (bytes = 0) => {
  if (!bytes) return '0 KB';
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(2)} MB`;
};

const ComparePage = () => {
  const navigate = useNavigate();

  const [documents, setDocuments] = useState([]);
  const [selectedDocIds, setSelectedDocIds] = useState([]);
  const [query, setQuery] = useState('');
  const [isLoadingDocs, setIsLoadingDocs] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const loadDocuments = async () => {
      setIsLoadingDocs(true);
      setError('');
      try {
        const payload = await fetchDocuments();
        const docs = Array.isArray(payload?.documents) ? payload.documents : [];
        setDocuments(docs);
      } catch (err) {
        setError(err.message || 'Failed to load documents');
      } finally {
        setIsLoadingDocs(false);
      }
    };

    loadDocuments();
  }, []);

  const sortedDocuments = useMemo(() => {
    return [...documents].sort((a, b) => {
      const aTime = new Date(a.uploaded_at || 0).getTime();
      const bTime = new Date(b.uploaded_at || 0).getTime();
      return bTime - aTime;
    });
  }, [documents]);

  const canCompare = query.trim().length > 0 && selectedDocIds.length >= 2;

  const toggleSelection = (documentId) => {
    setSelectedDocIds((prev) => {
      if (prev.includes(documentId)) {
        return prev.filter((id) => id !== documentId);
      }
      return [...prev, documentId];
    });
  };

  const handleCompare = (event) => {
    event.preventDefault();
    if (!canCompare) return;

    navigate('/compare/view', {
      state: {
        docIds: selectedDocIds,
        query: query.trim(),
      },
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-sky-50">
      <header className="sticky top-0 z-20 border-b border-orange-100 bg-white/95 backdrop-blur-sm">
        <div className="mx-auto max-w-6xl px-4 py-4 sm:px-6">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-orange-600">Compare Mode</p>
              <h1 className="text-2xl font-black text-slate-800">Cross-Document Comparison</h1>
            </div>

            <Link
              to="/chat"
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
            >
              Back to Chat
            </Link>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
        <form onSubmit={handleCompare} className="space-y-6">
          <section className="rounded-2xl border border-orange-100 bg-white shadow-sm">
            <div className="border-b border-orange-100 px-5 py-4">
              <h2 className="text-lg font-bold text-slate-800">1. Select Documents (2+)</h2>
              <p className="mt-1 text-sm text-slate-600">
                Choose at least two files to compare. V1 displays the first two selected documents side by side.
              </p>
            </div>

            <div className="p-5">
              {isLoadingDocs ? (
                <div className="flex items-center gap-2 text-sm text-slate-600">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading documents...
                </div>
              ) : null}

              {!isLoadingDocs && error ? (
                <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>
              ) : null}

              {!isLoadingDocs && !error && sortedDocuments.length === 0 ? (
                <p className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                  No documents found. Upload files in chat mode first.
                </p>
              ) : null}

              {!isLoadingDocs && sortedDocuments.length > 0 ? (
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                  {sortedDocuments.map((doc) => {
                    const isChecked = selectedDocIds.includes(doc.document_id);
                    return (
                      <label
                        key={doc.document_id}
                        className={`flex cursor-pointer items-start gap-3 rounded-xl border px-4 py-3 transition ${
                          isChecked
                            ? 'border-orange-400 bg-orange-50 shadow-sm'
                            : 'border-slate-200 bg-white hover:border-orange-200'
                        }`}
                      >
                        <input
                          type="checkbox"
                          className="mt-1 h-4 w-4 rounded border-slate-300 text-orange-600 focus:ring-orange-500"
                          checked={isChecked}
                          onChange={() => toggleSelection(doc.document_id)}
                        />

                        <div className="min-w-0 flex-1">
                          <div className="flex items-start gap-2">
                            <FileText className="mt-0.5 h-4 w-4 flex-shrink-0 text-orange-500" />
                            <p className="truncate text-sm font-semibold text-slate-800">{doc.file_name}</p>
                          </div>
                          <p className="mt-1 text-xs text-slate-500">{formatFileSize(doc.file_size)}</p>
                        </div>
                      </label>
                    );
                  })}
                </div>
              ) : null}
            </div>
          </section>

          <section className="rounded-2xl border border-sky-100 bg-white shadow-sm">
            <div className="border-b border-sky-100 px-5 py-4">
              <h2 className="text-lg font-bold text-slate-800">2. Enter Comparison Query</h2>
              <p className="mt-1 text-sm text-slate-600">Examples: compare achievements, compare backend experience</p>
            </div>

            <div className="p-5">
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="compare achievements"
                rows={3}
                className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm text-slate-800 outline-none transition focus:border-orange-400 focus:ring-2 focus:ring-orange-100"
              />
            </div>
          </section>

          <section className="flex items-center justify-between rounded-2xl border border-slate-200 bg-white px-5 py-4 shadow-sm">
            <p className="text-sm text-slate-600">
              Selected: <span className="font-semibold text-slate-800">{selectedDocIds.length}</span> documents
            </p>

            <button
              type="submit"
              disabled={!canCompare}
              className="inline-flex items-center gap-2 rounded-xl bg-orange-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-orange-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              <ArrowRightLeft className="h-4 w-4" />
              Compare
            </button>
          </section>
        </form>
      </main>
    </div>
  );
};

export default ComparePage;