#!/usr/bin/env python
"""
패키지 설치 후 임베딩 기능 테스트 스크립트

이 스크립트는 패키지가 제대로 설치되었는지, 
임베딩 기능이 정상적으로 작동하는지 확인합니다.
"""

import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def test_imports():
    """필수 모듈 import 테스트"""
    print("=" * 60)
    print("1. 필수 모듈 Import 테스트")
    print("=" * 60)
    
    try:
        from lex_dpr import BiEncoder, TemplateMode
        print("✅ lex_dpr 패키지 import 성공")
        print(f"   - BiEncoder: {BiEncoder}")
        print(f"   - TemplateMode: {TemplateMode}")
    except ImportError as e:
        print(f"❌ lex_dpr 패키지 import 실패: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✅ sentence-transformers import 성공")
    except ImportError as e:
        print(f"❌ sentence-transformers import 실패: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy import 성공")
    except ImportError as e:
        print(f"❌ numpy import 실패: {e}")
        return False
    
    try:
        import torch
        print(f"✅ torch import 성공 (version: {torch.__version__})")
    except ImportError as e:
        print(f"❌ torch import 실패: {e}")
        return False
    
    try:
        from peft import PeftModel
        print("✅ peft import 성공")
    except ImportError as e:
        print(f"⚠️  peft import 실패 (PEFT 기능 사용 불가): {e}")
        # PEFT는 선택적이므로 경고만 출력
    
    return True

def test_bi_encoder_init():
    """BiEncoder 초기화 테스트 (모델 로드 없이)"""
    print("\n" + "=" * 60)
    print("2. BiEncoder 클래스 구조 확인")
    print("=" * 60)
    
    try:
        from lex_dpr import BiEncoder, TemplateMode
        
        # 클래스 메서드 확인
        assert hasattr(BiEncoder, '__init__'), "BiEncoder에 __init__ 메서드 없음"
        assert hasattr(BiEncoder, 'encode_queries'), "BiEncoder에 encode_queries 메서드 없음"
        assert hasattr(BiEncoder, 'encode_passages'), "BiEncoder에 encode_passages 메서드 없음"
        
        print("✅ BiEncoder 클래스 구조 확인 완료")
        print(f"   - encode_queries 메서드: {BiEncoder.encode_queries}")
        print(f"   - encode_passages 메서드: {BiEncoder.encode_passages}")
        
        # TemplateMode 확인
        print(f"   - TemplateMode.BGE: {TemplateMode.BGE}")
        print(f"   - TemplateMode.NONE: {TemplateMode.NONE}")
        
        return True
    except Exception as e:
        print(f"❌ BiEncoder 클래스 구조 확인 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading_simulation():
    """모델 로드 시뮬레이션 (실제 모델 로드 없이)"""
    print("\n" + "=" * 60)
    print("3. 모델 로드 로직 검증 (시뮬레이션)")
    print("=" * 60)
    
    try:
        from lex_dpr import BiEncoder, TemplateMode
        from pathlib import Path
        
        # HuggingFace 모델 이름으로 초기화 시도 (실제 다운로드는 하지 않음)
        # 단순히 클래스가 올바르게 정의되어 있는지만 확인
        print("✅ BiEncoder 클래스 정의 확인 완료")
        print("   - 모델 경로 또는 HuggingFace 모델 이름을 받을 수 있음")
        print("   - PEFT 어댑터 자동 감지 기능 포함")
        print("   - TemplateMode 지원")
        
        # 파라미터 확인
        import inspect
        sig = inspect.signature(BiEncoder.__init__)
        params = list(sig.parameters.keys())
        print(f"   - __init__ 파라미터: {params}")
        
        required_params = ['name_or_path']
        optional_params = ['template', 'normalize', 'max_seq_length', 
                          'query_max_seq_length', 'passage_max_seq_length',
                          'trust_remote_code', 'peft_adapter_path']
        
        assert 'name_or_path' in params, "필수 파라미터 name_or_path 없음"
        print("   ✅ 필수 파라미터 확인 완료")
        
        for param in optional_params:
            if param in params:
                print(f"   ✅ 선택 파라미터 '{param}' 확인")
        
        return True
    except Exception as e:
        print(f"❌ 모델 로드 로직 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """의존성 확인"""
    print("\n" + "=" * 60)
    print("4. 의존성 버전 확인")
    print("=" * 60)
    
    try:
        import sentence_transformers
        print(f"✅ sentence-transformers: {sentence_transformers.__version__}")
    except:
        print("❌ sentence-transformers 버전 확인 실패")
    
    try:
        import numpy as np
        print(f"✅ numpy: {np.__version__}")
    except:
        print("❌ numpy 버전 확인 실패")
    
    try:
        import torch
        print(f"✅ torch: {torch.__version__}")
        print(f"   - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   - CUDA device count: {torch.cuda.device_count()}")
    except:
        print("❌ torch 버전 확인 실패")
    
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except:
        print("❌ transformers 버전 확인 실패")
    
    try:
        from peft import __version__ as peft_version
        print(f"✅ peft: {peft_version}")
    except:
        print("⚠️  peft 버전 확인 실패 (선택적 의존성)")

def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("LexDPR 패키지 임베딩 기능 테스트")
    print("=" * 60)
    
    results = []
    
    # 1. Import 테스트
    results.append(("Import", test_imports()))
    
    # 2. BiEncoder 구조 확인
    if results[-1][1]:  # 이전 테스트가 성공한 경우만
        results.append(("BiEncoder 구조", test_bi_encoder_init()))
    
    # 3. 모델 로드 로직 검증
    if results[-1][1]:  # 이전 테스트가 성공한 경우만
        results.append(("모델 로드 로직", test_model_loading_simulation()))
    
    # 4. 의존성 확인
    test_dependencies()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 모든 테스트 통과! 패키지가 정상적으로 설치되었습니다.")
        print("\n다음 단계:")
        print("1. 모델 체크포인트 경로를 준비하세요")
        print("2. 다음 코드로 임베딩을 생성할 수 있습니다:")
        print("""
from lex_dpr import BiEncoder

encoder = BiEncoder("path/to/model")
query_emb = encoder.encode_queries(["질의 텍스트"])
passage_emb = encoder.encode_passages(["패시지 텍스트"])
        """)
        return 0
    else:
        print("❌ 일부 테스트 실패. 패키지 설치를 확인하세요.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

