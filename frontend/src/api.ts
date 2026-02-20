import axios from 'axios';

const API_BASE_URL = '/api';

export interface Prediction {
  label: string;
  score: number;
  description?: string;
}

export interface PredictionResult {
  predictions: Prediction[];
  scores: Record<string, number>;
  metadata?: Record<string, any>;
}

export interface AnalysisResponse {
  request_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  filename: string;
  result?: PredictionResult;
  error?: string;
  processing_time_ms: number;
  created_at: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  ai_engine_connected: boolean;
  temp_dir_exists: boolean;
}

// AI Engine specific interfaces
export interface AIHealthResponse {
  status: 'healthy' | 'unhealthy' | 'unreachable';
  latency_ms: number;
  engine_url: string;
  details?: Record<string, any>;
  error?: string;
  request_id: string;
}

export interface AIAnalysisResponse {
  success: boolean;
  job_id?: string;
  outputs?: Record<string, any>;
  error?: string;
  processing_time_ms: number;
  request_id: string;
  filename: string;
}

export interface AIEngineInfoResponse {
  url: string;
  timeout: number;
  health_timeout: number;
  max_retries: number;
  allowed_extensions: string[];
}

// Docker control interfaces
export interface DockerStatusResponse {
  docker_installed: boolean;
  docker_running: boolean;
  container_exists: boolean;
  container_running: boolean;
  container_id?: string;
  gpu_available: boolean;
  error?: string;
}

export interface StartEngineResponse {
  success: boolean;
  message?: string;
  error?: string;
  container_id?: string;
  port?: number;
  gpu_enabled?: boolean;
  health?: Record<string, any>;
  warning?: string;
}

export interface StopEngineResponse {
  success: boolean;
  message?: string;
  error?: string;
}

export interface ContainerLogsResponse {
  success: boolean;
  logs?: string;
  error?: string;
}

export interface DiagnosticTestResult {
  name: string;
  status: 'pass' | 'fail' | 'warning' | 'skip';
  message: string;
  details?: Record<string, any>;
  duration_ms: number;
}

export interface DiagnosticsResponse {
  overall_status: 'pass' | 'fail' | 'warning';
  tests: DiagnosticTestResult[];
  timestamp: string;
  platform_info: Record<string, string>;
}

// ECG Analysis interfaces
export interface ECGModelInfo {
  name: string;
  description: string;
  category: string;
  priority: number;
  triggers?: string[];
}

export interface ECGModelsListResponse {
  models: Record<string, ECGModelInfo>;
  default_screening: string[];
  categories: string[];
}

export interface ECGPredictResponse {
  success: boolean;
  model_id: string;
  model_name: string;
  result?: Record<string, any>;
  error?: string;
  processing_time_ms: number;
  request_id: string;
}

export interface ECGModelResult {
  model_id: string;
  model_name: string;
  success: boolean;
  result?: Record<string, any>;
  error?: string;
  processing_time_ms: number;
}

export interface ECGScreeningSummary {
  flagged: string[];
  warnings: string[];
  recommended_next_steps: string[];
}

export interface ECGScreeningResponse {
  success: boolean;
  summary: ECGScreeningSummary;
  results: Record<string, ECGModelResult>;
  meta: {
    policy: string;
    models_requested: string[];
    models_executed: string[];
    total_time_ms: number;
    filename: string;
    error?: string;
  };
  request_id: string;
}

export type AnalysisMode = 'single' | 'screening';
export type ScreeningPolicy = 'all' | 'cascaded' | 'parallel';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analyzePdf = async (file: File): Promise<AnalysisResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<AnalysisResponse>('/analyze', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
};

// AI Engine API functions
export const checkAIHealth = async (): Promise<AIHealthResponse> => {
  const response = await api.get<AIHealthResponse>('/ai/health');
  return response.data;
};

export const analyzeWithAI = async (file: File): Promise<AIAnalysisResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<AIAnalysisResponse>('/ai/analyze', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 120000, // 2 minutes for AI processing
  });

  return response.data;
};

export const getAIEngineInfo = async (): Promise<AIEngineInfoResponse> => {
  const response = await api.get<AIEngineInfoResponse>('/ai/info');
  return response.data;
};

// Docker control API functions
export const getDockerStatus = async (): Promise<DockerStatusResponse> => {
  const response = await api.get<DockerStatusResponse>('/docker/status');
  return response.data;
};

export const startDockerEngine = async (image?: string, workspacePath?: string): Promise<StartEngineResponse> => {
  // Build request body, only including defined values
  const body: Record<string, string> = {};
  if (image) body.image = image;
  if (workspacePath) body.workspace_path = workspacePath;

  const response = await api.post<StartEngineResponse>('/docker/start', body);
  return response.data;
};

export const stopDockerEngine = async (): Promise<StopEngineResponse> => {
  const response = await api.post<StopEngineResponse>('/docker/stop');
  return response.data;
};

export const getContainerLogs = async (lines: number = 100): Promise<ContainerLogsResponse> => {
  const response = await api.get<ContainerLogsResponse>('/docker/logs', {
    params: { lines },
  });
  return response.data;
};

export const runDiagnostics = async (): Promise<DiagnosticsResponse> => {
  const response = await api.get<DiagnosticsResponse>('/docker/diagnostics');
  return response.data;
};

// ECG Analysis API functions
export const getECGModels = async (): Promise<ECGModelsListResponse> => {
  const response = await api.get<ECGModelsListResponse>('/ecg/models');
  return response.data;
};

export const predictSingleModel = async (
  file: File,
  modelId: string
): Promise<ECGPredictResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_id', modelId);

  const response = await api.post<ECGPredictResponse>('/ecg/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 120000,
  });

  return response.data;
};

export const screenECG = async (
  file: File,
  models?: string[],
  policy: ScreeningPolicy = 'all'
): Promise<ECGScreeningResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  if (models && models.length > 0) {
    formData.append('models', models.join(','));
  }
  formData.append('policy', policy);

  const response = await api.post<ECGScreeningResponse>('/ecg/screen', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 300000, // 5 minutes for full screening
  });

  return response.data;
};

// Preprocessing & Inference interfaces
export interface PreprocessingRequest {
  data_path: string;
  output_folder?: string;
  ecg_signals_path?: string;
  preprocessing_folder?: string;
  batch_size?: number;
  n_workers?: number;
  device?: string;
}

export interface PreprocessingResponse {
  success: boolean;
  message?: string;
  error?: string;
  output?: string;
}

export interface InferenceRequest {
  data_path: string;
  output_folder?: string;
  ecg_signals_path?: string;
  preprocessing_folder?: string;
  batch_size?: number;
  device?: string;
}

export interface InferenceResponse {
  success: boolean;
  message?: string;
  error?: string;
  output?: string;
}

// Preprocessing API function
export const runPreprocessing = async (request: PreprocessingRequest): Promise<PreprocessingResponse> => {
  const response = await api.post<PreprocessingResponse>('/docker/preprocessing', request, {
    timeout: 600000, // 10 minutes for preprocessing
  });
  return response.data;
};

// Inference API function
export const runInference = async (request: InferenceRequest): Promise<InferenceResponse> => {
  const response = await api.post<InferenceResponse>('/docker/inference', request, {
    timeout: 600000, // 10 minutes for inference
  });
  return response.data;
};

// ECG File Upload interfaces
export interface UploadECGResponse {
  success: boolean;
  message?: string;
  error?: string;
  ecg_filename?: string;
  csv_filename?: string;
  patient_id?: string;
}

export interface WorkspaceFilesResponse {
  success: boolean;
  ecg_files: string[];
  csv_files: string[];
  preprocessing_files: string[];
  output_files: string[];
  error?: string;
}

// Upload ECG file and generate CSV
export const uploadECGFile = async (
  file: File,
  workspacePath: string,
  patientId?: string
): Promise<UploadECGResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('workspace_path', workspacePath);
  if (patientId) {
    formData.append('patient_id', patientId);
  }

  const response = await api.post<UploadECGResponse>('/docker/upload-ecg', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

// List workspace files
export const getWorkspaceFiles = async (workspacePath: string): Promise<WorkspaceFilesResponse> => {
  const response = await api.get<WorkspaceFilesResponse>('/docker/workspace-files', {
    params: { workspace_path: workspacePath },
  });
  return response.data;
};

// Full Pipeline interfaces
export interface FullPipelineResponse {
  success: boolean;
  message?: string;
  error?: string;
  step?: string;
  preprocessing_output?: string;
  analysis_output?: string;
  result_files: string[];
  output_folder?: string;
}

// Run full pipeline with CSV file
export const runFullPipeline = async (
  file: File,
  useGpu: boolean = false,
  useWcr: boolean = false,
  useEfficientnet: boolean = true,
  batchSize: number = 1,
  patientId?: string
): Promise<FullPipelineResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('use_gpu', String(useGpu));
  formData.append('use_wcr', String(useWcr));
  formData.append('use_efficientnet', String(useEfficientnet));
  formData.append('batch_size', String(batchSize));
  if (patientId) {
    formData.append('patient_id', patientId);
  }

  const response = await api.post<FullPipelineResponse>('/docker/full-pipeline', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 1200000, // 20 minutes for full pipeline
  });
  return response.data;
};

// Analyze single ECG XML file (auto preprocessing + analysis)
export const analyzeECGFile = async (
  file: File,
  useGpu: boolean = false,
  useWcr: boolean = false,
  useEfficientnet: boolean = true,
  patientId?: string
): Promise<FullPipelineResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('use_gpu', String(useGpu));
  formData.append('use_wcr', String(useWcr));
  formData.append('use_efficientnet', String(useEfficientnet));
  if (patientId) {
    formData.append('patient_id', patientId);
  }

  const response = await api.post<FullPipelineResponse>('/docker/analyze-ecg', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 1200000, // 20 minutes for full pipeline
  });
  return response.data;
};

// ===================== Workspace Configuration =====================

export interface ConfigResponse {
  workspace_path: string;
  workspace_exists: boolean;
  subdirectories: Record<string, boolean>;
}

export const getConfig = async (): Promise<ConfigResponse> => {
  const response = await api.get<ConfigResponse>('/config');
  return response.data;
};

export const updateConfig = async (workspacePath: string): Promise<ConfigResponse> => {
  const response = await api.put<ConfigResponse>('/config', {
    workspace_path: workspacePath,
  });
  return response.data;
};

// ===================== Full ECG Analysis with Model Selection =====================

export interface ECGModelAvailable {
  id: string;
  name: string;
  architecture: 'efficientnet' | 'wcr';
  type: 'multi_label' | 'binary';
  description?: string;
}

export interface ECGModelsAvailableResponse {
  models: ECGModelAvailable[];
  default_selection: string[];
}

export interface ECGDiagnosis {
  name: string;
  probability: number;
  threshold: number;
  status: 'normal' | 'borderline' | 'abnormal';
  category: string;
}

export interface ECGModelAnalysisResult {
  model_id: string;
  model_name: string;
  model_type: string;
  architecture: string;
  success: boolean;
  error?: string;
  diagnoses: ECGDiagnosis[];
  by_category: Record<string, ECGDiagnosis[]>;
}

export interface ECGCriticalFinding {
  diagnosis: string;
  probability: number;
  model: string;
  category?: string;
}

export interface ECGAnalysisSummary {
  overall_status: 'normal' | 'borderline' | 'abnormal';
  total_abnormal: number;
  total_borderline: number;
  critical_findings: ECGCriticalFinding[];
}

export interface ECGFileFormatInfo {
  original_format: string;
  original_encoding: string;
  conversions_applied: string[];
  conversion_notes?: string;
}

export interface FullECGAnalysisResponse {
  success: boolean;
  patient_id: string;
  ecg_filename: string;
  file_format_info?: ECGFileFormatInfo;
  models_executed: string[];
  results: Record<string, ECGModelAnalysisResult>;
  summary: ECGAnalysisSummary;
  warnings: string[];
  processing_time_ms: number;
  error?: string;
}

// Get available ECG models
export const getAvailableECGModels = async (): Promise<ECGModelsAvailableResponse> => {
  const response = await api.get<ECGModelsAvailableResponse>('/ecg/available-models');
  return response.data;
};

// Run full ECG analysis with selected models
export const runFullECGAnalysis = async (
  file: File,
  models: string[] = ['all'],
  useGpu: boolean = false,
  patientId?: string
): Promise<FullECGAnalysisResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('models', models.join(','));
  formData.append('use_gpu', String(useGpu));
  if (patientId) {
    formData.append('patient_id', patientId);
  }

  const response = await api.post<FullECGAnalysisResponse>('/ecg/full-analysis', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 1800000, // 30 minutes for full analysis with all models
  });

  // If the main analysis returned no diagnoses, try to parse existing results
  const hasResults = Object.values(response.data.results || {}).some(
    r => r.diagnoses && r.diagnoses.length > 0
  );

  if (!hasResults && response.data.success) {
    console.log('No diagnoses in response, trying parse-existing endpoint...');
    try {
      const existingResponse = await api.get<FullECGAnalysisResponse>('/ecg/parse-existing');
      if (existingResponse.data.success) {
        // Merge the patient_id and models_executed from original response
        return {
          ...existingResponse.data,
          patient_id: response.data.patient_id,
          ecg_filename: response.data.ecg_filename || existingResponse.data.ecg_filename,
          models_executed: response.data.models_executed,
          processing_time_ms: response.data.processing_time_ms,
        };
      }
    } catch (e) {
      console.error('Failed to parse existing results:', e);
    }
  }

  return response.data;
};

// ===================== Batch ECG Analysis =====================

export interface BatchECGResult {
  index: number;
  filename: string;
  patient_id: string;
  success: boolean;
  result?: FullECGAnalysisResponse;
  error?: string;
}

export interface BatchECGAnalysisResponse {
  success: boolean;
  total_files: number;
  successful: number;
  failed: number;
  results: BatchECGResult[];
  total_processing_time_ms: number;
  models_used: string[];
}

// Run batch ECG analysis on multiple files
export const runBatchECGAnalysis = async (
  files: File[],
  models: string[] = ['all'],
  useGpu: boolean = true
): Promise<BatchECGAnalysisResponse> => {
  const formData = new FormData();

  // Append all files
  files.forEach(file => {
    formData.append('files', file);
  });

  formData.append('models', models.join(','));
  formData.append('use_gpu', String(useGpu));

  const response = await api.post<BatchECGAnalysisResponse>('/ecg/batch-analysis', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 3600000, // 60 minutes for batch processing
  });

  return response.data;
};