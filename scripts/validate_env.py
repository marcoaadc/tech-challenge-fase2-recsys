"""Valida o ambiente de desenvolvimento do projeto de recomendacao.

Checa a versao do Python, a disponibilidade das dependencias principais
e a presenca dos arquivos de configuracao esperados, imprimindo um
relatorio OK/FAIL. Usa apenas a stdlib; as dependencias do projeto sao
importadas de forma opcional.

Uso:
    python scripts/validate_env.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

MIN_PYTHON: tuple[int, int] = (3, 10)
REQUIRED_MODULES: tuple[str, ...] = (
    "torch",
    "sklearn",
    "pandas",
    "mlflow",
    "dvc",
    "pydantic_settings",
)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

CheckResult = tuple[str, bool, str]


def check_python_version() -> CheckResult:
    """Verifica se o interpretador atende a versao minima exigida.

    Returns:
        Tupla (nome do check, passou, detalhe).
    """
    current = sys.version_info[:2]
    detail = f"encontrado Python {current[0]}.{current[1]}"
    ok = current >= MIN_PYTHON
    if not ok:
        detail += f", requerido >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    return ("python >= 3.10", ok, detail)


def check_import(module_name: str) -> CheckResult:
    """Tenta importar um modulo e reporta o resultado.

    Args:
        module_name: Nome do modulo a importar (ex.: "torch").

    Returns:
        Tupla (nome do check, passou, detalhe).
    """
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # ImportError e erros de inicializacao
        return (f"import {module_name}", False, str(exc))
    version = getattr(module, "__version__", "versao desconhecida")
    return (f"import {module_name}", True, str(version))


def check_params_file() -> CheckResult:
    """Verifica a existencia do arquivo de parametros do pipeline.

    Returns:
        Tupla (nome do check, passou, detalhe).
    """
    path = PROJECT_ROOT / "configs" / "params.yaml"
    if path.is_file():
        return ("configs/params.yaml", True, "encontrado")
    return ("configs/params.yaml", False, f"nao encontrado em {path}")


def check_env_file() -> CheckResult:
    """Verifica a existencia do arquivo .env (ou do template .env.example).

    A ausencia de .env nao reprova o check quando .env.example existe:
    os valores padrao cobrem o uso local, entao emite apenas um aviso.

    Returns:
        Tupla (nome do check, passou, detalhe).
    """
    env = PROJECT_ROOT / ".env"
    example = PROJECT_ROOT / ".env.example"
    if env.is_file():
        return (".env", True, "encontrado")
    if example.is_file():
        detail = "AVISO: .env ausente; copie .env.example para .env se precisar customizar"
        return (".env", True, detail)
    return (".env", False, "nem .env nem .env.example encontrados")


def run_checks() -> list[CheckResult]:
    """Executa todos os checks de ambiente na ordem de relatorio.

    Returns:
        Lista de resultados individuais.
    """
    results: list[CheckResult] = [check_python_version()]
    results.extend(check_import(name) for name in REQUIRED_MODULES)
    results.append(check_params_file())
    results.append(check_env_file())
    return results


def print_report(results: list[CheckResult]) -> bool:
    """Imprime o relatorio formatado e resume o resultado geral.

    Args:
        results: Resultados produzidos por ``run_checks``.

    Returns:
        True se todos os checks passaram, False caso contrario.
    """
    print("=" * 60)
    print("Validacao de ambiente - recsys")
    print("=" * 60)
    for name, ok, detail in results:
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {name:<28} {detail}")
    passed = all(ok for _, ok, _ in results)
    total = len(results)
    good = sum(1 for _, ok, _ in results if ok)
    print("-" * 60)
    print(f"Resultado: {good}/{total} checks passaram")
    return passed


def main() -> int:
    """Ponto de entrada do script.

    Returns:
        0 se o ambiente esta valido, 1 caso contrario.
    """
    results = run_checks()
    return 0 if print_report(results) else 1


if __name__ == "__main__":
    sys.exit(main())
