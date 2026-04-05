/**
 * PDFViewer Component
 * Bbox-only highlight rendering driven by backend-provided word boxes.
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from 'lucide-react';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

const normalizeBoxes = (boxes) => {
	if (!Array.isArray(boxes)) return [];

	return boxes
		.map((box) => {
			if (!Array.isArray(box) || box.length < 4) return null;
			const parsed = box.slice(0, 4).map((value) => Number(value));
			if (!parsed.every((value) => Number.isFinite(value))) return null;
			const [x1, y1, x2, y2] = parsed;
			if (x2 <= x1 || y2 <= y1) return null;
			return [x1, y1, x2, y2];
		})
		.filter(Boolean);
};

const mergeAdjacentBoxes = (boxes) => {
	if (!Array.isArray(boxes) || boxes.length <= 1) return boxes || [];

	const sorted = [...boxes].sort((a, b) => {
		if (Math.abs(a[1] - b[1]) > 1) return a[1] - b[1];
		return a[0] - b[0];
	});

	const merged = [];
	const lineTolerance = 3;
	const gapTolerance = 8;

	sorted.forEach((box) => {
		if (merged.length === 0) {
			merged.push([...box]);
			return;
		}

		const last = merged[merged.length - 1];
		const sameLine = Math.abs(last[1] - box[1]) <= lineTolerance && Math.abs(last[3] - box[3]) <= lineTolerance;
		const smallGap = box[0] - last[2] <= gapTolerance;

		if (sameLine && smallGap) {
			last[0] = Math.min(last[0], box[0]);
			last[1] = Math.min(last[1], box[1]);
			last[2] = Math.max(last[2], box[2]);
			last[3] = Math.max(last[3], box[3]);
			return;
		}

		merged.push([...box]);
	});

	return merged;
};

const HighlightLayer = ({ boxes }) => {
	if (!boxes.length) return null;

	return (
		<div className="absolute inset-0 pointer-events-none z-20">
			{boxes.map((box, index) => {
				const [x1, y1, x2, y2] = box;

				return (
					<div
						key={`${index}-${x1}-${y1}`}
						style={{
							position: 'absolute',
							left: x1,
							top: y1,
							width: x2 - x1,
							height: y2 - y1,
							backgroundColor: 'rgba(255, 220, 0, 0.45)',
							borderRadius: 2,
						}}
					/>
				);
			})}
		</div>
	);
};

const PDFViewer = ({ file, activePage, activeBoxes, highlightTrigger }) => {
	const [numPages, setNumPages] = useState(null);
	const [pageNumber, setPageNumber] = useState(1);
	const [scale, setScale] = useState(1.0);
	const [loading, setLoading] = useState(true);
	const [pdfPageSize, setPdfPageSize] = useState({ width: 0, height: 0 });
	const [renderedPageSize, setRenderedPageSize] = useState({ width: 0, height: 0 });

	const containerRef = useRef(null);
	const pageWrapperRef = useRef(null);

	const pdfSource = file.documentId
		? `http://localhost:8000/upload/documents/${file.documentId}/file`
		: file;

	useEffect(() => {
		if (activePage && Number.isFinite(Number(activePage))) {
			setPageNumber(Math.max(1, Number(activePage)));
		}
	}, [activePage, highlightTrigger]);

	useEffect(() => {
		if (numPages && pageNumber > numPages) {
			setPageNumber(numPages);
		}
	}, [numPages, pageNumber]);

	useEffect(() => {
		const count = Array.isArray(activeBoxes) ? activeBoxes.length : 0;
		console.log('HIGHLIGHT BOXES:', count);
	}, [activeBoxes, highlightTrigger]);

	const updateRenderedPageSize = () => {
		const wrapper = pageWrapperRef.current;
		if (!wrapper) return;

		const canvas = wrapper.querySelector('canvas');
		if (!canvas) return;

		const rect = canvas.getBoundingClientRect();
		if (rect.width <= 0 || rect.height <= 0) return;

		setRenderedPageSize({
			width: rect.width,
			height: rect.height,
		});
	};

	useEffect(() => {
		if (typeof ResizeObserver === 'undefined') return undefined;

		const wrapper = pageWrapperRef.current;
		if (!wrapper) return undefined;

		const observer = new ResizeObserver(() => {
			updateRenderedPageSize();
		});

		observer.observe(wrapper);
		return () => observer.disconnect();
	}, [pageNumber, scale]);

	const scaledBoxes = useMemo(() => {
		const boxes = normalizeBoxes(activeBoxes);
		if (!boxes.length) return [];

		const { width: originalWidth, height: originalHeight } = pdfPageSize;
		const { width: renderedWidth, height: renderedHeight } = renderedPageSize;
		if (!originalWidth || !originalHeight || !renderedWidth || !renderedHeight) {
			return [];
		}

		const scaleX = renderedWidth / originalWidth;
		const scaleY = renderedHeight / originalHeight;

		const scaled = boxes.map(([x1, y1, x2, y2]) => [
			x1 * scaleX,
			y1 * scaleY,
			x2 * scaleX,
			y2 * scaleY,
		]);

		return mergeAdjacentBoxes(scaled);
	}, [activeBoxes, pdfPageSize, renderedPageSize]);

	useEffect(() => {
		if (!scaledBoxes.length) return;

		const container = containerRef.current;
		const wrapper = pageWrapperRef.current;
		if (!container || !wrapper) return;

		const [, y1] = scaledBoxes[0];
		const containerRect = container.getBoundingClientRect();
		const wrapperRect = wrapper.getBoundingClientRect();

		const targetTop = wrapperRect.top - containerRect.top + container.scrollTop + y1;
		const targetScrollTop = Math.max(0, targetTop - container.clientHeight / 2);
		container.scrollTo({ top: targetScrollTop, behavior: 'smooth' });
	}, [scaledBoxes, highlightTrigger, pageNumber]);

	const onDocumentLoadSuccess = ({ numPages: pages }) => {
		setNumPages(pages);
		setLoading(false);
	};

	const onDocumentLoadError = (error) => {
		console.error('Error loading PDF:', error);
		setLoading(false);
	};

	const goToPrevPage = () => setPageNumber((prev) => Math.max(prev - 1, 1));
	const goToNextPage = () => setPageNumber((prev) => Math.min(prev + 1, numPages || prev + 1));
	const zoomIn = () => setScale((prev) => Math.min(prev + 0.2, 2.5));
	const zoomOut = () => setScale((prev) => Math.max(prev - 0.2, 0.5));

	return (
		<div className="flex flex-col h-full bg-gray-50">
			<div className="flex items-center justify-between px-4 py-3 bg-white border-b border-gray-200 shadow-sm">
				<div className="flex items-center space-x-2">
					<button
						onClick={goToPrevPage}
						disabled={pageNumber <= 1}
						className="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition"
					>
						<ChevronLeft className="w-5 h-5" />
					</button>

					<div className="px-3 py-1 bg-gray-100 rounded-lg text-sm font-medium">
						<span className="text-gray-700">{pageNumber}</span>
						<span className="text-gray-400"> / {numPages || '...'}</span>
					</div>

					<button
						onClick={goToNextPage}
						disabled={pageNumber >= (numPages || 0)}
						className="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition"
					>
						<ChevronRight className="w-5 h-5" />
					</button>
				</div>

				<div className="flex items-center space-x-2">
					<button
						onClick={zoomOut}
						disabled={scale <= 0.5}
						className="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition"
					>
						<ZoomOut className="w-5 h-5" />
					</button>

					<div className="px-3 py-1 bg-gray-100 rounded-lg text-sm font-medium text-gray-700 min-w-[4rem] text-center">
						{Math.round(scale * 100)}%
					</div>

					<button
						onClick={zoomIn}
						disabled={scale >= 2.5}
						className="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition"
					>
						<ZoomIn className="w-5 h-5" />
					</button>
				</div>
			</div>

			<div className="flex-1 overflow-auto p-6" ref={containerRef}>
				<div className="flex justify-center">
					{loading && (
						<div className="flex items-center justify-center py-20">
							<div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
						</div>
					)}

					<Document
						file={pdfSource}
						onLoadSuccess={onDocumentLoadSuccess}
						onLoadError={onDocumentLoadError}
						loading=""
						className="shadow-lg"
					>
						<div className="relative inline-block" ref={pageWrapperRef}>
							<Page
								pageNumber={pageNumber}
								scale={scale}
								renderTextLayer={true}
								renderAnnotationLayer={true}
								onLoadSuccess={(page) => {
									try {
										const viewport = page.getViewport({ scale: 1 });
										setPdfPageSize({
											width: viewport.width || 0,
											height: viewport.height || 0,
										});
									} catch {
										setPdfPageSize({ width: 0, height: 0 });
									}
								}}
								onRenderSuccess={() => {
									updateRenderedPageSize();
								}}
								className="bg-white shadow-xl rounded-lg overflow-hidden"
							/>

							<HighlightLayer boxes={scaledBoxes} />
						</div>
					</Document>
				</div>
			</div>
		</div>
	);
};

export default PDFViewer;