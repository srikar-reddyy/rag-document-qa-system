import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { ChevronLeft, ChevronRight, Search, ZoomIn, ZoomOut } from 'lucide-react';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

const normalizeForMatch = (value) =>
  String(value || '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();

const buildHighlightTargets = (highlights) => {
  const seenText = new Set();
  const seenPages = new Set();
  const textTargets = [];
  const pageHints = [];

  for (const item of highlights || []) {
    const rawText = typeof item === 'string' ? item : item?.text || '';
    const rawPage = typeof item === 'object' ? Number(item?.page) : NaN;

    if (Number.isFinite(rawPage) && rawPage > 0 && !seenPages.has(rawPage)) {
      seenPages.add(rawPage);
      pageHints.push(rawPage);
    }

    const normalized = normalizeForMatch(rawText);
    if (!normalized) continue;
    const bounded = normalized.slice(0, 280);
    if (seenText.has(bounded)) continue;
    seenText.add(bounded);
    textTargets.push(bounded);
    if (textTargets.length >= 18) break;
  }

  return {
    textTargets,
    pageHints: pageHints.sort((a, b) => a - b),
  };
};

const ComparePDFViewer = ({ fileUrl, highlights = [], highlightColor = 'rgba(250, 204, 21, 0.24)' }) => {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(1.0);
  const [matchedPages, setMatchedPages] = useState([]);
  const [isSearchingMatches, setIsSearchingMatches] = useState(false);

  const searchTokenRef = useRef(0);

  const { textTargets, pageHints } = useMemo(() => buildHighlightTargets(highlights), [highlights]);
  const matchedPageSet = useMemo(() => new Set(matchedPages), [matchedPages]);

  useEffect(() => {
    setNumPages(null);
    setPageNumber(1);
    setMatchedPages(pageHints);
  }, [fileUrl, pageHints]);

  const findMatchedPages = async (pdf) => {
    const token = Date.now();
    searchTokenRef.current = token;

    if (!pdf?.numPages) {
      setMatchedPages(pageHints);
      return;
    }

    if (!textTargets.length) {
      setMatchedPages(pageHints);
      if (pageHints.length > 0) {
        setPageNumber((current) => (pageHints.includes(current) ? current : pageHints[0]));
      }
      return;
    }

    setIsSearchingMatches(true);
    const nextMatches = [...pageHints];

    try {
      for (let i = 1; i <= pdf.numPages; i += 1) {
        if (searchTokenRef.current !== token) {
          return;
        }

        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        const pageText = normalizeForMatch(content.items.map((item) => item.str || '').join(' '));

        if (!pageText) continue;

        const hasMatch = textTargets.some((target) => pageText.includes(target));
        if (hasMatch) {
          if (!nextMatches.includes(i)) {
            nextMatches.push(i);
          }
        }
      }

      if (searchTokenRef.current === token) {
        nextMatches.sort((a, b) => a - b);
        setMatchedPages(nextMatches);
        if (nextMatches.length > 0) {
          setPageNumber((current) => (nextMatches.includes(current) ? current : nextMatches[0]));
        }
      }
    } finally {
      if (searchTokenRef.current === token) {
        setIsSearchingMatches(false);
      }
    }
  };

  const handleLoadSuccess = (pdf) => {
    setNumPages(pdf.numPages);
    findMatchedPages(pdf);
  };

  const goToPrevPage = () => setPageNumber((prev) => Math.max(1, prev - 1));
  const goToNextPage = () => setPageNumber((prev) => Math.min(numPages || prev, prev + 1));
  const zoomIn = () => setScale((prev) => Math.min(2.4, prev + 0.2));
  const zoomOut = () => setScale((prev) => Math.max(0.6, prev - 0.2));

  const goToPrevMatchedPage = () => {
    if (!matchedPages.length) return;
    const currentIdx = matchedPages.findIndex((p) => p === pageNumber);
    if (currentIdx <= 0) {
      setPageNumber(matchedPages[matchedPages.length - 1]);
      return;
    }
    setPageNumber(matchedPages[currentIdx - 1]);
  };

  const goToNextMatchedPage = () => {
    if (!matchedPages.length) return;
    const currentIdx = matchedPages.findIndex((p) => p === pageNumber);
    if (currentIdx < 0 || currentIdx >= matchedPages.length - 1) {
      setPageNumber(matchedPages[0]);
      return;
    }
    setPageNumber(matchedPages[currentIdx + 1]);
  };

  return (
    <div className="flex h-full min-h-[22rem] flex-col overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 bg-white px-3 py-2">
        <div className="flex items-center gap-1">
          <button
            onClick={goToPrevPage}
            disabled={pageNumber <= 1}
            className="rounded-md p-1.5 text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>

          <span className="min-w-[5rem] text-center text-xs font-semibold text-slate-700">
            Page {pageNumber} / {numPages || '...'}
          </span>

          <button
            onClick={goToNextPage}
            disabled={numPages ? pageNumber >= numPages : true}
            className="rounded-md p-1.5 text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={zoomOut}
            disabled={scale <= 0.6}
            className="rounded-md p-1.5 text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
          >
            <ZoomOut className="h-4 w-4" />
          </button>

          <span className="min-w-[3rem] text-center text-xs font-semibold text-slate-700">{Math.round(scale * 100)}%</span>

          <button
            onClick={zoomIn}
            disabled={scale >= 2.4}
            className="rounded-md p-1.5 text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="flex items-center justify-between border-b border-slate-200 bg-amber-50/70 px-3 py-2">
        <div className="flex items-center gap-2 text-xs text-amber-900">
          <Search className="h-4 w-4" />
          {isSearchingMatches ? 'Scanning pages for matching sections...' : `Matched pages: ${matchedPages.join(', ') || 'none'}`}
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={goToPrevMatchedPage}
            disabled={matchedPages.length === 0}
            className="rounded-md border border-amber-200 bg-white px-2 py-1 text-xs font-semibold text-amber-900 disabled:cursor-not-allowed disabled:text-amber-300"
          >
            Prev Match
          </button>
          <button
            onClick={goToNextMatchedPage}
            disabled={matchedPages.length === 0}
            className="rounded-md border border-amber-200 bg-white px-2 py-1 text-xs font-semibold text-amber-900 disabled:cursor-not-allowed disabled:text-amber-300"
          >
            Next Match
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4">
        <div className="flex justify-center">
          <Document file={fileUrl} onLoadSuccess={handleLoadSuccess} loading="Loading PDF..." className="shadow-md">
            <div className="relative inline-block">
              <Page
                pageNumber={pageNumber}
                scale={scale}
                renderTextLayer={true}
                renderAnnotationLayer={true}
                className="overflow-hidden rounded-lg border border-slate-200 bg-white"
              />

              {matchedPageSet.has(pageNumber) ? (
                <div
                  className="pointer-events-none absolute inset-0 rounded-lg border-2 border-amber-300"
                  style={{ backgroundColor: highlightColor }}
                />
              ) : null}
            </div>
          </Document>
        </div>
      </div>
    </div>
  );
};

export default ComparePDFViewer;