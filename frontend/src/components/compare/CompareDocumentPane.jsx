import React from 'react';
import { ExternalLink, FileText } from 'lucide-react';

import ComparePDFViewer from './ComparePDFViewer';
import { getDocumentFileUrl } from '../../services/api';

const CompareDocumentPane = ({ title, documentData, tint = 'amber' }) => {
  if (!documentData) {
    return (
      <article className="flex h-full min-h-[28rem] items-center justify-center rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-500">
        No document data available.
      </article>
    );
  }

  const docName = documentData.doc_name || 'Unknown document';
  const isPdf = docName.toLowerCase().endsWith('.pdf');
  const fileUrl = documentData.doc_id ? getDocumentFileUrl(documentData.doc_id) : '';
  const borderClass = tint === 'sky' ? 'border-sky-200' : 'border-amber-200';
  const badgeClass = tint === 'sky' ? 'bg-sky-100 text-sky-800' : 'bg-amber-100 text-amber-800';

  return (
    <article className={`flex h-full flex-col overflow-hidden rounded-2xl border ${borderClass} bg-white shadow-sm`}>
      <header className="border-b border-slate-200 px-4 py-3">
        <div className="flex items-center justify-between gap-2">
          <span className={`rounded-full px-2.5 py-1 text-xs font-bold uppercase tracking-wide ${badgeClass}`}>{title}</span>
          {fileUrl ? (
            <a
              href={fileUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-xs font-semibold text-slate-600 hover:text-slate-900"
            >
              Open
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          ) : null}
        </div>

        <p className="mt-2 truncate text-sm font-semibold text-slate-800">{docName}</p>
      </header>

      <div className="flex-1 p-4">
        {isPdf && fileUrl ? (
          <ComparePDFViewer
            fileUrl={fileUrl}
            highlights={documentData.highlights || []}
            highlightColor={tint === 'sky' ? 'rgba(56, 189, 248, 0.24)' : 'rgba(250, 204, 21, 0.24)'}
          />
        ) : (
          <div className="flex h-full min-h-[22rem] flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50 px-4 text-center">
            <FileText className="mb-2 h-5 w-5 text-slate-400" />
            <p className="text-sm font-semibold text-slate-700">Viewer is optimized for PDF documents in compare mode.</p>
            <p className="mt-1 text-xs text-slate-500">Open the file directly to inspect non-PDF formats.</p>
          </div>
        )}
      </div>
    </article>
  );
};

export default CompareDocumentPane;