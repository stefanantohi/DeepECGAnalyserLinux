import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';

interface ECGSignalData {
  success: boolean;
  filename: string;
  leads: string[];
  samples_per_lead: number;
  original_samples: number;
  sample_rate: number;
  duration_seconds: number;
  data: Record<string, number[]>;
  error?: string;
}

interface ECGViewerProps {
  filename?: string;
  onClose?: () => void;
}

// Standard 12-lead ECG layout (4 columns x 3 rows)
const ECG_LAYOUT = [
  ['I', 'aVR', 'V1', 'V4'],
  ['II', 'aVL', 'V2', 'V5'],
  ['III', 'aVF', 'V3', 'V6'],
];

// ECG paper settings (standard clinical: 25mm/s, 10mm/mV)
const PAPER_SPEED_MM_PER_SEC = 25; // 25 mm/s paper speed
const SMALL_SQUARE_MM = 1; // 1mm small squares (0.04s, 0.1mV)
const LARGE_SQUARE_MM = 5; // 5mm large squares (0.2s, 0.5mV)

// Display settings
const PIXELS_PER_MM = 3; // Scale factor for screen display
const SMALL_SQUARE_PX = SMALL_SQUARE_MM * PIXELS_PER_MM;
const LARGE_SQUARE_PX = LARGE_SQUARE_MM * PIXELS_PER_MM;

// Colors for classic ECG paper
const COLORS = {
  background: '#FFF8F0', // Cream/off-white paper
  smallGrid: '#FFD4D4', // Light pink for small squares
  largeGrid: '#FFB0B0', // Darker pink for large squares
  trace: '#000000', // Black trace
  label: '#333333', // Dark gray labels
};

const ECGViewer: React.FC<ECGViewerProps> = ({ filename = 'ecg.xml', onClose }) => {
  const [signalData, setSignalData] = useState<ECGSignalData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [displaySeconds, setDisplaySeconds] = useState(2.5); // Seconds per lead to display
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const fetchSignalData = async () => {
      try {
        setLoading(true);
        setError(null);
        setSignalData(null); // Clear old data
        // Add cache-busting timestamp to avoid browser caching
        const cacheBuster = Date.now();
        const response = await axios.get<ECGSignalData>(
          `/api/ecg/signal-data?filename=${encodeURIComponent(filename)}&_t=${cacheBuster}`
        );
        if (response.data.success) {
          setSignalData(response.data);
        } else {
          setError(response.data.error || 'Failed to load ECG data');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load ECG data');
      } finally {
        setLoading(false);
      }
    };

    fetchSignalData();
  }, [filename]);

  useEffect(() => {
    if (!signalData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Calculate dimensions
    const numRows = 3;
    const numCols = 4;
    const leadWidth = displaySeconds * PAPER_SPEED_MM_PER_SEC * PIXELS_PER_MM;
    const leadHeight = 30 * PIXELS_PER_MM; // 30mm per lead row (3mV range)
    const labelWidth = 30; // Space for lead labels
    const marginTop = 10;
    const marginBottom = 10;
    const rowSpacing = 5;
    const rhythmStripSpacing = 15; // Extra space before rhythm strip

    const totalWidth = labelWidth + (numCols * leadWidth);

    // Calculate rhythm strip position
    const rhythmY = marginTop + (numRows * (leadHeight + rowSpacing)) + rhythmStripSpacing;
    const rhythmHeight = leadHeight;

    // Calculate TOTAL height including rhythm strip BEFORE setting canvas size
    const totalHeight = rhythmY + rhythmHeight + marginBottom;

    // Set canvas size ONCE (changing size clears canvas!)
    canvas.width = totalWidth;
    canvas.height = totalHeight;

    // Fill background for entire canvas
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, totalWidth, totalHeight);

    // Draw grid and traces for each lead (4x3 grid)
    for (let row = 0; row < numRows; row++) {
      for (let col = 0; col < numCols; col++) {
        const leadName = ECG_LAYOUT[row][col];
        const leadData = signalData.data[leadName];

        const x = labelWidth + (col * leadWidth);
        const y = marginTop + (row * (leadHeight + rowSpacing));

        // Draw grid for this lead area
        drawGrid(ctx, x, y, leadWidth, leadHeight);

        // Draw lead label
        ctx.fillStyle = COLORS.label;
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(leadName, x - 5, y + leadHeight / 2);

        // Draw ECG trace
        if (leadData) {
          drawTrace(ctx, leadData, x, y, leadWidth, leadHeight, signalData.sample_rate, displaySeconds);
        }
      }
    }

    // Draw rhythm strip (Lead II) at the bottom
    const rhythmWidth = totalWidth - labelWidth;

    // Label for rhythm strip
    ctx.fillStyle = COLORS.label;
    ctx.font = 'bold 12px Arial';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('II', labelWidth - 5, rhythmY + rhythmHeight / 2);

    // Draw grid for rhythm strip (full width)
    drawGrid(ctx, labelWidth, rhythmY, rhythmWidth, rhythmHeight);

    // Draw full rhythm trace (Lead II with longer duration)
    const rhythmData = signalData.data['II'];
    if (rhythmData) {
      const rhythmSeconds = Math.min(signalData.duration_seconds, 10); // Up to 10 seconds
      drawTrace(ctx, rhythmData, labelWidth, rhythmY, rhythmWidth, rhythmHeight, signalData.sample_rate, rhythmSeconds);
    }

  }, [signalData, displaySeconds]);

  const drawGrid = (
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    width: number,
    height: number
  ) => {
    // Draw small squares
    ctx.strokeStyle = COLORS.smallGrid;
    ctx.lineWidth = 0.5;

    // Vertical lines (small)
    for (let i = 0; i <= width; i += SMALL_SQUARE_PX) {
      ctx.beginPath();
      ctx.moveTo(x + i, y);
      ctx.lineTo(x + i, y + height);
      ctx.stroke();
    }

    // Horizontal lines (small)
    for (let i = 0; i <= height; i += SMALL_SQUARE_PX) {
      ctx.beginPath();
      ctx.moveTo(x, y + i);
      ctx.lineTo(x + width, y + i);
      ctx.stroke();
    }

    // Draw large squares
    ctx.strokeStyle = COLORS.largeGrid;
    ctx.lineWidth = 1;

    // Vertical lines (large)
    for (let i = 0; i <= width; i += LARGE_SQUARE_PX) {
      ctx.beginPath();
      ctx.moveTo(x + i, y);
      ctx.lineTo(x + i, y + height);
      ctx.stroke();
    }

    // Horizontal lines (large)
    for (let i = 0; i <= height; i += LARGE_SQUARE_PX) {
      ctx.beginPath();
      ctx.moveTo(x, y + i);
      ctx.lineTo(x + width, y + i);
      ctx.stroke();
    }

    // Draw border
    ctx.strokeStyle = COLORS.largeGrid;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, width, height);
  };

  const drawTrace = (
    ctx: CanvasRenderingContext2D,
    data: number[],
    x: number,
    y: number,
    width: number,
    height: number,
    sampleRate: number,
    durationSeconds: number
  ) => {
    // Calculate samples to display
    const samplesToShow = Math.min(Math.floor(sampleRate * durationSeconds), data.length);
    const samplesPerPixel = samplesToShow / width;

    // Scaling: height is 30mm = 3mV range, centered
    const mvPerPixel = 3 / height;
    const centerY = y + height / 2;

    ctx.strokeStyle = COLORS.trace;
    ctx.lineWidth = 1.2;
    ctx.beginPath();

    let started = false;
    for (let px = 0; px < width; px++) {
      const sampleIndex = Math.floor(px * samplesPerPixel);
      if (sampleIndex >= data.length) break;

      const mv = data[sampleIndex]; // Data is already in mV
      const traceY = centerY - (mv / mvPerPixel); // Invert Y (up is positive)

      // Clamp to bounds
      const clampedY = Math.max(y, Math.min(y + height, traceY));

      if (!started) {
        ctx.moveTo(x + px, clampedY);
        started = true;
      } else {
        ctx.lineTo(x + px, clampedY);
      }
    }
    ctx.stroke();

    // Draw calibration pulse (1mV, 0.2s) at the start
    drawCalibrationPulse(ctx, x + 2, centerY, height);
  };

  const drawCalibrationPulse = (
    ctx: CanvasRenderingContext2D,
    x: number,
    centerY: number,
    height: number
  ) => {
    // 1mV calibration pulse
    const mvHeight = height / 3; // 1mV in the 3mV range
    const pulseWidth = 10;

    ctx.strokeStyle = COLORS.trace;
    ctx.lineWidth = 1.2;
    ctx.beginPath();

    // Draw calibration square wave
    ctx.moveTo(x, centerY);
    ctx.lineTo(x, centerY - mvHeight);
    ctx.lineTo(x + pulseWidth, centerY - mvHeight);
    ctx.lineTo(x + pulseWidth, centerY);

    ctx.stroke();
  };

  if (loading) {
    return (
      <div className="bg-gray-900 rounded-xl p-6 text-center">
        <div className="animate-spin h-8 w-8 border-4 border-green-500 border-t-transparent rounded-full mx-auto mb-4"></div>
        <p className="text-gray-300">Chargement du tracé ECG...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-500 rounded-xl p-6 text-center">
        <p className="text-red-400 font-medium mb-2">Erreur de chargement de l'ECG</p>
        <p className="text-red-300 text-sm">{error}</p>
        <p className="text-gray-500 text-xs mt-2">Fichier recherché: {filename}</p>
      </div>
    );
  }

  if (!signalData) return null;

  return (
    <div className="bg-gray-900 rounded-xl p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-white flex items-center gap-2">
            ECG 12 Dérivations
          </h3>
          <p className="text-sm text-gray-400">
            {signalData.duration_seconds.toFixed(1)}s @ {signalData.sample_rate}Hz |
            25mm/s | 10mm/mV
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={displaySeconds}
            onChange={(e) => setDisplaySeconds(parseFloat(e.target.value))}
            className="bg-gray-700 text-white px-3 py-2 rounded-lg text-sm"
          >
            <option value={2.5}>2.5s par dérivation</option>
            <option value={5}>5s par dérivation</option>
            <option value={10}>10s par dérivation</option>
          </select>
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 rounded-lg bg-gray-700 text-gray-300 hover:bg-gray-600"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* ECG Canvas */}
      <div className="overflow-x-auto bg-white rounded-lg p-2">
        <canvas
          ref={canvasRef}
          className="min-w-full"
          style={{ imageRendering: 'crisp-edges' }}
        />
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between text-xs text-gray-500 border-t border-gray-700 pt-3">
        <div className="flex items-center gap-4">
          <span>1 petit carré = 0.04s (40ms) | 0.1mV</span>
          <span>1 grand carré = 0.2s (200ms) | 0.5mV</span>
        </div>
        <span>Format: Standard 12 dérivations + tracé rythmique DII</span>
      </div>
    </div>
  );
};

export default ECGViewer;
