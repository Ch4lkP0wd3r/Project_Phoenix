"use client";

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FileUpload } from '@/components/FileUpload';
import { ScoreDisplay } from '@/components/ScoreDisplay';
import { Shield, Lock, History, BarChart3, Fingerprint } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE_URL = "http://localhost:8000";

export default function Home() {
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'processing' | 'completed' | 'failed'>('idle');
  const [results, setResults] = useState<any>(null);

  const handleUpload = async (file: File) => {
    setStatus('processing');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/analyze`, formData);
      setAnalysisId(response.data.analysis_id);
    } catch (error) {
      console.error("Upload failed", error);
      setStatus('failed');
    }
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (analysisId && status === 'processing') {
      interval = setInterval(async () => {
        try {
          const response = await axios.get(`${API_BASE_URL}/score/${analysisId}`);
          if (response.data.status === 'completed') {
            setResults(response.data);
            setStatus('completed');
            clearInterval(interval);
          } else if (response.data.status === 'failed') {
            setStatus('failed');
            clearInterval(interval);
          }
        } catch (error) {
          console.error("Polling failed", error);
          setStatus('failed');
          clearInterval(interval);
        }
      }, 3000);
    }

    return () => clearInterval(interval);
  }, [analysisId, status]);

  const handleDownloadReport = () => {
    if (analysisId) {
      window.open(`${API_BASE_URL}/report/${analysisId}`, '_blank');
    }
  };

  const reset = () => {
    setAnalysisId(null);
    setStatus('idle');
    setResults(null);
  };

  return (
    <main className="min-h-screen pb-20">
      {/* Header */}
      <nav className="p-8 flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center space-x-2">
          <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
            <Shield className="text-white" size={24} />
          </div>
          <span className="text-2xl font-bold tracking-tight">PHOENIX</span>
        </div>
        <div className="hidden md:flex items-center space-x-8 text-sm font-medium text-neutral-400">
          <a href="#" className="hover:text-white transition-colors">Forensics</a>
          <a href="#" className="hover:text-white transition-colors">Legal 65B</a>
          <a href="#" className="hover:text-white transition-colors">Verify Report</a>
        </div>
        <button className="px-5 py-2 rounded-full border border-neutral-800 text-sm hover:bg-white hover:text-black transition-all">
          Connect HSM
        </button>
      </nav>

      {/* Hero Content */}
      <section className="pt-20 pb-12 text-center max-w-4xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="text-5xl md:text-7xl font-bold mb-6 tracking-tight">
            Media <span className="gradient-text">Authenticity</span> at Scale
          </h1>
          <p className="text-lg text-neutral-400 mb-12 max-w-2xl mx-auto leading-relaxed">
            Project Phoenix uses next-generation AI and cryptographic signatures to detect deepfakes and generate legally admissible evidence bundles under Section 65B Bharatiya Sakshya Adhiniyam.
          </p>
        </motion.div>

        {/* Feature Pills */}
        <div className="flex flex-wrap items-center justify-center gap-4 mb-20">
          {[
            { icon: Lock, label: "AES-256 Encrypted" },
            { icon: Fingerprint, label: "PGP Signed" },
            { icon: BarChart3, label: "Xception Analysis" },
            { icon: History, label: "Chain of Custody" }
          ].map((f, i) => (
            <div key={i} className="flex items-center space-x-2 px-4 py-2 rounded-full bg-neutral-900 border border-neutral-800 text-xs text-neutral-400">
              <f.icon size={14} />
              <span>{f.label}</span>
            </div>
          ))}
        </div>

        <AnimatePresence mode="wait">
          {status === 'idle' || status === 'processing' ? (
            <motion.div
              key="upload"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
            >
              <FileUpload onUpload={handleUpload} isProcessing={status === 'processing'} />
            </motion.div>
          ) : status === 'completed' && results ? (
            <motion.div
              key="results"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <ScoreDisplay
                score={results.score}
                visualScore={results.visual_score}
                audioScore={results.audio_score}
                onDownloadReport={handleDownloadReport}
              />
              <button
                onClick={reset}
                className="mt-8 text-neutral-500 hover:text-white text-sm underline underline-offset-4"
              >
                Analyze another file
              </button>
            </motion.div>
          ) : (
            <div className="text-rose-400">
              <p>Analysis failed. Please try again with a different file.</p>
              <button onClick={reset} className="mt-4 px-6 py-2 bg-neutral-800 rounded-lg">Retry</button>
            </div>
          )}
        </AnimatePresence>
      </section>

      {/* Decorative Background Elements */}
      <div className="fixed top-0 left-0 w-full h-full pointer-events-none -z-10 overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[120px]" />
      </div>
    </main>
  );
}
