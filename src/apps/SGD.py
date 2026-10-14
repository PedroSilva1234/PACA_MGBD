import math
import os
import glob
import logging
import subprocess
import sys
import re
from typing import Dict, List, Tuple, Optional, Any

from src.apps.base import BaseApp
from src.code_parser import parse_code
from src.hash_utils import gerar_hash_codigo_logico
from src.database.variant_tracker import load_executed_variants
from src.utils.file_utils import short_hash, copy_file
from src.execution.simulation import run_spike_simulation

class SgdApp(BaseApp):
    """Aplicação SGD (Gradient Descent) herdando de BaseApp."""
    
    CONFIG: Dict[str, Any] = {
        "original_file": "data/applications/SGD/src/SGD.c",
        "input_file_for_variants": "data/applications/SGD/src/SGD.c",
        "source_pattern": "SGD_*.c",
        "exe_prefix": "sgd_",
        "output_suffix": ".data",
        "time_suffix": ".time",
        "prof5_suffix": ".prof5",
        "prof5_model": "data/models/APPROX_1.json",
        "operations_map": {'*': 'FMULX', '+': 'FADDX', '-': 'FSUBX'},
        "optimization_level": "-O3",
        # SGD não usa arquivo externo de input (dados estão hardcoded no .c)
        "train_data_input": "", 
    }
    
    REQUIRED_CONFIG_KEYS = [
        "original_file",
        "input_file_for_variants",
        "operations_map",
        "exe_prefix"
    ]
    
    def __init__(self):
        super().__init__()
    
    def get_config(self) -> Dict[str, Any]:
        return self.CONFIG
    

    
    
    def prepare_environment(self, base_config: Dict) -> bool:
        config = self._merge_config(base_config)
        approx_source = config.get("approx_file", "data/reference/approx.h")
        os.makedirs(config["input_dir"], exist_ok=True)
        # Copia o header de aproximações para a pasta de trabalho
        if os.path.exists(approx_source):
            return copy_file(approx_source, config["input_dir"])
        return True
    
    def generate_variants(self, base_config: Dict) -> bool:
        config = self._merge_config(base_config)
        try:
            input_path = os.path.abspath(config["input_file_for_variants"])
            output_path = os.path.abspath(config["input_dir"])
            executados_path = os.path.abspath(config["executed_variants_file"])
            
            cmd = [
                sys.executable, "src/gera_variantes.py",
                "--input", input_path,
                "--output", output_path,
                "--executados", executados_path,
                "--strategy", "all"
            ]
            
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=1800)
            
            pattern = os.path.join(config["input_dir"], "SGD_*.c")
            generated_files = glob.glob(pattern)
            return len(generated_files) > 0
        except Exception as e:
            logging.error(f"Erro na geração do SGD: {e}")
            return False

    def find_variants_to_simulate(self, base_config: Dict) -> Tuple[List[Tuple[str, str]], Dict]:
        # Implementação padrão (semelhante ao JMeint, mas adaptada para SGD.c)
        config = self._merge_config(base_config)
        executed_variants = load_executed_variants(config["executed_variants_file"])
        variants_to_simulate = []
        
        original_path = config["original_file"]
        with open(original_path, "r") as f:
            original_lines = f.readlines()
        _, __, original_physical_to_logical = parse_code(original_path)
        
        original_hash = gerar_hash_codigo_logico(original_lines, original_physical_to_logical)
        
        if original_hash not in executed_variants:
            variants_to_simulate.append((original_path, original_hash))
        
        variant_pattern = os.path.join(config["input_dir"], config["source_pattern"])
        variant_files = glob.glob(variant_pattern)
        
        for variant_file_path in variant_files:
            if os.path.abspath(variant_file_path) == os.path.abspath(original_path):
                continue
            with open(variant_file_path, "r") as f:
                variant_lines = f.readlines()
            variant_hash = gerar_hash_codigo_logico(variant_lines, original_physical_to_logical)
            
            if variant_hash not in executed_variants:
                variants_to_simulate.append((variant_file_path, variant_hash))
                
        return variants_to_simulate, original_physical_to_logical


    def _compile_sgd_variant(self, variant_cpp: str, output_hash: str, config: Dict, status_monitor) -> Tuple[bool, Optional[str]]:
        """Compilação simples e direta de um único arquivo .c para RISC-V."""
        is_original = os.path.abspath(variant_cpp) == os.path.abspath(config["original_file"])
        variant_id = "original" if is_original else short_hash(output_hash)
        status_monitor.update_status(variant_id, "Compilando SGD")
        
        exe_prefix = config.get("exe_prefix", "sgd_")
        executables_dir = config["executables_dir"]
        optimization = config.get("optimization_level", "-O3")
        
        exe_file = os.path.join(executables_dir, f"{exe_prefix}{output_hash}")
        
        # Como SGD.c é em C puro, usamos o gcc (ou g++ se houver compatibilidade)
        compile_cmd = [
            "riscv32-unknown-elf-gcc", "-march=rv32imafdcv", optimization,
            "-I", config["input_dir"],
            "-I", "data/reference",
            variant_cpp, "-o", exe_file, "-lm"
        ]
        try:
            subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
            os.chmod(exe_file, 0o755)
            return True, exe_file
        except subprocess.CalledProcessError as e:
            logging.error(f"[{variant_id}] Erro compilação SGD: {e.stderr.strip()}")
            status_monitor.update_status(variant_id, "Erro Compilação SGD")
            return False, None

    def simulate_variant(self, variant_file: str, variant_hash: str, base_config: Dict, status_monitor, only_spike: bool = False) -> Tuple[Optional[str], Optional[Dict]]:
        config = self._merge_config(base_config)
        is_original = os.path.abspath(variant_file) == os.path.abspath(config["original_file"])
        variant_id = "original" if is_original else short_hash(variant_hash)
        
        exe_prefix = config.get("exe_prefix", "sgd_")
        spike_output_file = os.path.join(config["outputs_dir"], f"{exe_prefix}{variant_hash}{config['output_suffix']}")
        spike_log_file = os.path.join(config["logs_dir"], f"{exe_prefix}{variant_hash}.log")
        
        compiled_ok, exe_file = self._compile_sgd_variant(variant_file, variant_hash, config, status_monitor)
        if not compiled_ok:
            return (None, None)
            
        sim_time = run_spike_simulation(
            exe_file, config["train_data_input"], spike_output_file,
            spike_log_file, variant_id, status_monitor
        )
        if sim_time is None:
            return (None, None)
            
        # RETORNO OFICIAL: Sem invenções, o C joga o resultado direto aqui!
        return spike_output_file, None


    #-------------------------------------------------------------------------


    def calculate_custom_error(self, reference_file: str, variant_file: str) -> Optional[float]:
        """Extrai os parâmetros (m e b) e calcula o erro absoluto médio (MAE)."""
        import re
        import logging

        def extrair_parametros(filepath):
            try:
                with open(filepath, 'r') as f:
                    conteudo = f.read()
                
                # Regex atualizado para aceitar nan e inf (ignorando maiúsculas/minúsculas)
                import re
                match = re.search(r"Reta com (?:FLOAT|DOUBLE)\s*:\s*y\s*=\s*([-+]?(?:\d*\.\d+|\d+|nan|inf))\s*x\s*\+\s*([-+]?(?:\d*\.\d+|\d+|nan|inf))", conteudo, re.IGNORECASE)
                if match:
                    return float(match.group(1)), float(match.group(2))
                return None, None
            except Exception:
                return None, None

        def calcular_rmse(m_ref, b_ref, m_var, b_var):
            """
            Calcula o RMSE (Root Mean Squared Error).
            Retorna o erro absoluto na mesma unidade dos dados preditos.
            """
            erro_quadratico_total = 0.0
            dados_x = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            for x in dados_x:
                y_ref = (m_ref * x) + b_ref        
                y_var = (m_var * x) + b_var
                
                diferenca = y_ref - y_var
                erro_quadratico_total += (diferenca * diferenca)
                
            mse = erro_quadratico_total / len(dados_x)  
            rmse = math.sqrt(mse)
            
            return rmse

        try:
            m_ref, b_ref = extrair_parametros(reference_file)
            m_var, b_var = extrair_parametros(variant_file)
            
            if m_ref is None:
                logging.error("[SGD Metric] Falha fatal: Nao foi possivel ler a referencia.")
                return 1.0
                
            import math
            # Se a variante explodiu para NaN ou Infinito, damos o erro máximo
            if m_var is None or math.isnan(m_var) or math.isinf(m_var) or math.isnan(b_var) or math.isinf(b_var):
                logging.warning(f"[SGD Metric] Gradiente explodiu (NaN/Inf) na variante.")
                return 1.0 # Penalidade máxima
                
            erro_m = abs(m_ref - m_var)
            erro_b = abs(b_ref - b_var)
            return (erro_m + erro_b) / 2.0
            
        except Exception as e:
            logging.error(f"[SGD Error] Exceção no cálculo: {e}")
            return None
# ==========================================
# Instância global e funções wrapper para compatibilidade
# com a arquitetura principal do run.py
# ==========================================
app = SgdApp()

def get_config(): return app.get_config()
def prepare_environment(base_config): return app.prepare_environment(base_config)
def generate_variants(base_config): return app.generate_variants(base_config)
def find_variants_to_simulate(base_config): return app.find_variants_to_simulate(base_config)
def simulate_variant(variant_file, variant_hash, base_config, status_monitor, only_spike=False):
    return app.simulate_variant(variant_file, variant_hash, base_config, status_monitor, only_spike)
def calculate_custom_error(reference_file, variant_file):
    return app.calculate_custom_error(reference_file, variant_file)