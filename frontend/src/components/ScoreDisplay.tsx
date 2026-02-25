"use client";

import React from 'react';
import { motion } from 'framer-motion';
import { ShieldCheck, ShieldAlert, Fingerprint, Download, FileJson, FileText } from 'lucide-react';

interface ScoreDisplayProps {
    score: number;
    visualScore?: number;
    audioScore?: number;
    onDownloadReport: () => void;
}

export const ScoreDisplay: React.FC<ScoreDisplayProps> = ({ score, visualScore, audioScore, onDownloadReport }) => {
    const getStatus = (s: number) => {
        if (s > 70) return { label: "High Integrity", color: "text-emerald-400", bg: "bg-emerald-400/10", icon: ShieldCheck };
        if (s > 40) return { label: "Suspicious", color: "text-amber-400", bg: "bg-amber-400/10", icon: ShieldAlert };
        return { label: "Manipulation Detected", color: "text-rose-400", bg: "bg-rose-400/10", icon: ShieldAlert };
    };

    const status = getStatus(score);
    const Icon = status.icon;

    return (
        <div className="w-full max-w-4xl mx-auto space-y-8 animate-in fade-in duration-700">
            <div className="glass-card p-12 flex flex-col items-center relative overflow-hidden">
                {/* Glow effect */}
                <div className={`absolute -top-24 -right-24 w-64 h-64 rounded-full blur-[100px] opacity-20 ${status.bg}`} />

                <div className={`mb-6 p-4 rounded-2xl ${status.bg} ${status.color}`}>
                    <Icon size={48} />
                </div>

                <h2 className="text-4xl font-bold mb-2">{score}%</h2>
                <p className={`text-xl font-medium ${status.color} mb-8`}>{status.label}</p>

                <div className="w-full h-2 bg-neutral-800 rounded-full overflow-hidden mb-12 max-w-md">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${score}%` }}
                        transition={{ duration: 1.5, ease: "easeOut" }}
                        className={`h-full ${status.bg.replace('/10', '')} ${status.color.replace('text', 'bg')}`}
                    />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
                    <div className="glass-card p-6 bg-neutral-900/50">
                        <p className="text-sm text-neutral-500 mb-1">Visual Authenticity</p>
                        <p className="text-2xl font-mono">{visualScore ?? 'N/A'}{visualScore ? '%' : ''}</p>
                    </div>
                    <div className="glass-card p-6 bg-neutral-900/50">
                        <p className="text-sm text-neutral-500 mb-1">Audio Authenticity</p>
                        <p className="text-2xl font-mono">{audioScore ?? 'N/A'}{audioScore ? '%' : ''}</p>
                    </div>
                </div>
            </div>

            <div className="glass-card p-8 border-blue-500/10">
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center space-x-3">
                        <div className="p-2 rounded bg-blue-500/10 text-blue-400">
                            <Fingerprint size={24} />
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold">Evidence Bundle</h3>
                            <p className="text-sm text-neutral-500">Section 65B Compliant Package</p>
                        </div>
                    </div>
                    <button
                        onClick={onDownloadReport}
                        className="flex items-center space-x-2 px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-500 transition-colors font-medium shadow-lg shadow-blue-500/20"
                    >
                        <Download size={18} />
                        <span>Download Bundle</span>
                    </button>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg bg-neutral-900/40 border border-neutral-800 flex items-center space-x-3">
                        <FileJson size={20} className="text-neutral-400" />
                        <span className="text-sm">JSON Manifest</span>
                    </div>
                    <div className="p-4 rounded-lg bg-neutral-900/40 border border-neutral-800 flex items-center space-x-3">
                        <FileText size={20} className="text-neutral-400" />
                        <span className="text-sm">PDF Report</span>
                    </div>
                    <div className="p-4 rounded-lg bg-neutral-900/40 border border-neutral-800 flex items-center space-x-3">
                        <ShieldCheck size={20} className="text-neutral-400" />
                        <span className="text-sm">PGP Signature</span>
                    </div>
                </div>
            </div>
        </div>
    );
};
