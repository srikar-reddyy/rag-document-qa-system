/**
 * Main App Component
 * Professional 3-panel layout with resizable panels, collapsible chat, and maximizable viewer
 * Architecture: Sidebar | Document Viewer | Chat Panel
 */

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
// Removed logo icon import
import useResizablePanels from './hooks/useResizablePanels';
import Sidebar from './components/Sidebar';
import ViewerPanel from './components/ViewerPanel';
import ChatPanel from './components/ChatPanel';
import Resizer from './components/Resizer';
import { fetchDocuments, deleteDocument as apiDeleteDocument, fetchChatHistory } from './services/api';
import { createInitialHighlightState, buildHighlightState } from './HighlightStore';

function App() {
  // Document state
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [showUpload, setShowUpload] = useState(true);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState([]); // NEW: Track selected documents for RAG
  
  // Chat state
  const [chatMessages, setChatMessages] = useState([]);
  const [isLoadingState, setIsLoadingState] = useState(true);
  const [activeHighlight, setActiveHighlight] = useState(createInitialHighlightState);

  // Resizable panels hook
  const {
    sidebarWidth,
    chatWidth,
    isChatCollapsed,
    isViewerMaximized,
    isResizing,
    handleMouseDown,
    toggleChatCollapse,
    toggleViewerMaximize,
    getViewerWidth,
  } = useResizablePanels(320, 420);

  // Load persisted state on mount
  useEffect(() => {
    const loadPersistedState = async () => {
      try {
        // Fetch documents
        const documentsData = await fetchDocuments();
        if (documentsData.documents && documentsData.documents.length > 0) {
          // Convert backend document format to file objects
          const files = documentsData.documents.map(doc => ({
            name: doc.file_name,
            documentId: doc.document_id,
            size: doc.file_size,
            uploadedAt: doc.uploaded_at,
            path: doc.file_path
          }));
          setUploadedFiles(files);
          setShowUpload(false);
        }

        // Fetch chat history
        const historyData = await fetchChatHistory();
        if (historyData.messages && historyData.messages.length > 0) {
          setChatMessages(historyData.messages);
        }

        console.log('State restored successfully');
      } catch (error) {
        console.error('Error loading persisted state:', error);
      } finally {
        setIsLoadingState(false);
      }
    };

    loadPersistedState();
  }, []);

  // File upload success handler
  const handleUploadSuccess = async (files, response) => {
    // Refresh document list from backend to get document IDs
    try {
      const documentsData = await fetchDocuments();
      if (documentsData.documents) {
        const filesList = documentsData.documents.map(doc => ({
          name: doc.file_name,
          documentId: doc.document_id,
          size: doc.file_size,
          uploadedAt: doc.uploaded_at,
          path: doc.file_path
        }));
        setUploadedFiles(filesList);
        
        if (!selectedFile && filesList.length > 0) {
          setSelectedFile(filesList[0]);
          setShowUpload(false);
        }
      }
    } catch (error) {
      console.error('Error refreshing documents:', error);
      // Fallback to local state if fetch fails
      setUploadedFiles(prev => [...prev, ...files]);
      if (!selectedFile && files.length > 0) {
        setSelectedFile(files[0]);
        setShowUpload(false);
      }
    }
  };

  // File selection handler
  const handleSelectFile = (file) => {
    setSelectedFile(file);
    setShowUpload(false);
  };

  // File deletion handler
  const handleDeleteFile = async (fileToDelete) => {
    try {
      // Delete from backend if we have a document ID
      if (fileToDelete.documentId) {
        await apiDeleteDocument(fileToDelete.documentId);
        console.log('Document deleted from backend');
        
        // Remove from selected documents if it was selected
        setSelectedDocumentIds(prev => prev.filter(id => id !== fileToDelete.documentId));
      }
      
      // Update local state
      setUploadedFiles(prev => prev.filter(f => f.name !== fileToDelete.name));
      
      if (selectedFile?.name === fileToDelete.name) {
        setSelectedFile(null);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
    }
  };

  // Document checkbox toggle handler
  const handleToggleDocument = (documentId) => {
    setSelectedDocumentIds(prev => {
      if (prev.includes(documentId)) {
        // Uncheck
        return prev.filter(id => id !== documentId);
      } else {
        // Check
        return [...prev, documentId];
      }
    });
  };

  const handleHighlightClick = (highlight) => {
    if (!highlight) return;

    const targetDocName = (highlight.doc_name || '').toLowerCase();
    const targetFile = uploadedFiles.find(
      (file) => (file.name || '').toLowerCase() === targetDocName
    );

    if (targetFile) {
      setSelectedFile(targetFile);
      setShowUpload(false);
    }

    setActiveHighlight(buildHighlightState(highlight));
  };

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-slate-50 to-slate-100 overflow-hidden">
      {/* Header */}
      <header className="flex-shrink-0 bg-white border-b border-gray-200 shadow-sm z-20">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div>
                <h1 className="text-xl font-bold text-gray-800">
                  Multi-Document Reasoning Engine
                </h1>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <Link
                to="/compare"
                className="px-3 py-1.5 text-sm font-semibold text-amber-800 bg-amber-100 hover:bg-amber-200 rounded-lg transition-colors"
              >
                Compare Mode
              </Link>
              
              <div className="text-xs text-gray-500">
                {uploadedFiles.length} {uploadedFiles.length === 1 ? 'document' : 'documents'}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content - 3 Panel Layout */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Left Sidebar - Document List */}
        <Sidebar
          width={sidebarWidth}
          isHidden={isViewerMaximized}
          uploadedFiles={uploadedFiles}
          selectedFile={selectedFile}
          onSelectFile={handleSelectFile}
          onDeleteFile={handleDeleteFile}
          onUploadSuccess={handleUploadSuccess}
          showUpload={showUpload}
          setShowUpload={setShowUpload}
          selectedDocumentIds={selectedDocumentIds}
          onToggleDocument={handleToggleDocument}
        />

        {/* Resizer - Sidebar/Viewer */}
        {!isViewerMaximized && (
          <Resizer
            onMouseDown={handleMouseDown('sidebar')}
            isResizing={isResizing === 'sidebar'}
          />
        )}

        {/* Center Panel - Document Viewer */}
        <ViewerPanel
          file={selectedFile}
          isMaximized={isViewerMaximized}
          onToggleMaximize={toggleViewerMaximize}
          highlightState={activeHighlight}
          style={{ width: getViewerWidth() }}
        />

        {/* Resizer - Viewer/Chat */}
        {!isViewerMaximized && !isChatCollapsed && (
          <Resizer
            onMouseDown={handleMouseDown('chat')}
            isResizing={isResizing === 'chat'}
          />
        )}

        {/* Right Panel - Chat */}
        <ChatPanel
          width={chatWidth}
          isCollapsed={isChatCollapsed}
          onToggleCollapse={toggleChatCollapse}
          isHidden={isViewerMaximized}
          initialMessages={chatMessages}
          isLoadingState={isLoadingState}
          selectedDocumentIds={selectedDocumentIds}
          onHighlightClick={handleHighlightClick}
        />
      </div>

      {/* Footer Status Bar */}
      {/* <footer className="flex-shrink-0 bg-white border-t border-gray-200 px-6 py-2">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center space-x-4">
          </div>
          <div className="flex items-center space-x-4">
            <span className={`flex items-center space-x-1 ${isResizing ? 'text-primary-600' : ''}`}>
              <div className={`w-1.5 h-1.5 rounded-full ${isResizing ? 'bg-primary-500 animate-pulse' : 'bg-gray-400'}`}></div>
              <span>{isResizing ? 'Resizing...' : 'Ready'}</span>
            </span>
            <span className="text-gray-300">|</span>
            <span>Phase 2: RAG + ChromaDB</span>
          </div>
        </div>
      </footer> */}
    </div>
  );
}

export default App;
