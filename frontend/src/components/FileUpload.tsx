"use client";

import React, { useState, useCallback } from 'react';
import { Upload, File, X, CheckCircle, ShieldAlert } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface FileUploadProps {
    onUpload: (file: File) => void;
    isProcessing: boolean;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onUpload, isProcessing }) => {
    const [dragActive, setDragActive] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [consent, setConsent] = useState(false);

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setSelectedFile(e.dataTransfer.files[0]);
        }
    }, []);

    const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            setSelectedFile(e.target.files[0]);
        }
    }, []);

    const clearFile = () => setSelectedFile(null);

    const handleSubmit = () => {
        if (selectedFile && consent) {
            onUpload(selectedFile);
        }
    };

    return (
        <div className="w-full max-w-2xl mx-auto space-y-6">
            <div
                className={`relative glass-card bg-neutral-900/40 p-12 border-2 border-dashed transition-all ${dragActive ? "border-blue-500 bg-blue-500/5" : "border-neutral-800"
                    } ${isProcessing ? "opacity-50 pointer-events-none" : ""}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    onChange={handleChange}
                    accept="video/*,audio/*,image/*"
                />

                <div className="flex flex-col items-center justify-center text-center space-y-4">
                    <div className="p-4 rounded-full bg-blue-500/10 text-blue-400">
                        <Upload size={32} />
                    </div>
                    <div>
                        <p className="text-xl font-medium">Drop your media here</p>
                        <p className="text-neutral-500 text-sm mt-1">Supports Video, Audio, and Image Forensic Analysis</p>
                    </div>
                </div>
            </div>

            <AnimatePresence>
                {selectedFile && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="glass-card p-4 flex items-center justify-between border-blue-500/20"
                    >
                        <div className="flex items-center space-x-3">
                            <div className="p-2 rounded bg-neutral-800">
                                <File size={20} className="text-blue-400" />
                            </div>
                            <div>
                                <p className="text-sm font-medium">{selectedFile.name}</p>
                                <p className="text-xs text-neutral-500">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                            </div>
                        </div>
                        <button onClick={clearFile} className="p-1 hover:bg-neutral-800 rounded">
                            <X size={18} className="text-neutral-400" />
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>

            {selectedFile && (
                <div className="space-y-4">
                    <label className="flex items-start space-x-3 cursor-pointer group">
                        <div className="pt-0.5">
                            <input
                                type="checkbox"
                                checked={consent}
                                onChange={(e) => setConsent(e.target.checked)}
                                className="w-4 h-4 rounded border-neutral-700 bg-neutral-900 text-blue-500 cursor-pointer focus:ring-0"
                            />
                        </div>
                        <span className="text-sm text-neutral-400 group-hover:text-neutral-300 transition-colors leading-relaxed">
                            I consent to the collection and analysis of this media for authenticity verification purposes. I understand that data will be deleted after report generation.
                        </span>
                    </label>

                    <button
                        onClick={handleSubmit}
                        disabled={!consent || isProcessing}
                        className="w-full py-4 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-neutral-800 disabled:text-neutral-600 font-semibold transition-all shadow-lg shadow-blue-500/20 active:scale-[0.98]"
                    >
                        {isProcessing ? "Analyzing Patterns..." : "Secure Analysis"}
                    </button>
                </div>
            )}

            <div className="flex items-center space-x-2 text-neutral-500 text-xs justify-center pt-4">
                <ShieldAlert size={14} />
                <span>Advisory tool — verify results before legal use.</span>
            </div>
        </div>
    );
};
