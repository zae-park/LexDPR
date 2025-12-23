---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:13091
- loss:MixedNegativesRankingLoss
base_model: intfloat/multilingual-e5-small
widget:
- source_sentence: 'Represent this sentence for searching relevant passages: 상호저축은행업감독규정의
    ''주택 관련 담보대출 등에 대한 리스크관리기준'' 별표 내용은 무엇인가?'
  sentences:
  - 'Represent this sentence for retrieving relevant passages: ① 시행령 제9조의2제1항제3호 및
    동조 제2항제4호의 규정에 의한 신용공여 한도 초과사유 또는 시행령 제9조의3제3호의 규정에 의한 신용공여 한도 초과기간의 연장사유의 인정
    또는 승인을 받고자 하는 상호저축은행은 금융위에 그 인정 또는 승인을 신청하여야 한다. ② 상호저축은행은 시행령 제9조의2에서 정하는 사유로
    인하여 법 제12조제1항에서 제3항까지의 규정에 의한 한도를 초과하거나 시행령 제9조의3의 규정에 따라 한도의 초과기간을 연장하는 경우에는
    한도초과일 또는 한도초과기간 연장일로부터 15일 이내에 감독원장에게 보고하여야 한다. ③ 제1항 및 제2항에서 정하는 사항 이외에 인정신청
    절차, 한도초과보고 절차 기타 필요한 사항은 감독원장이 정한다.'
  - 'Represent this sentence for retrieving relevant passages: ① 이 규정에서 사용하는 용어의 정의는
    법령이 정하는 바에 의한다. ② 이 규정에서 "동일계열기업"이라 함은 시행령 제30조제2항제5호에서 제8호까지에 해당하는 법인 등을 말한다.
    ③ 이 규정에서 "금융기관"이라 함은 금융위원회의 설치등에 관한 법률 제38조에 따라 금융감독원의 검사를 받는 기관을 말한다. ④ 이 규정에서
    "예비인가"라 함은 인가사항에 대한 사전심사 및 확실한 실행을 위하여 인가 이전에 예비적으로 행하여지는 행위로서 인가의 효력이 없는 것을 말한다.
    ⑤ 이 규정에서 "자기자본"이란 법 제2조제4호에 따른 자기자본을 말한다. 다만, 제14조, 제24조, 제42조, 제58조 및 별표 4에서
    "자기자본"이란 대차대조표상 자산총액에서 부채총액을 뺀 금액을 말한다.'
  - 'Represent this sentence for retrieving relevant passages: [별표 1]<신설 2005.12.29>

    인 가 절 차

    ┌───────────────────────────────────┐

    │                                                                      │

    │┌───────┐┌──────────┐                            │

    ││예비 인가단계 ││절차안내            │                            │

    ││              │└──────────┘                            │

    ││              │  ↓                                                │

    ││              │┌──────────┐                            │

    ││              ││예비인가 신청       │                            │

    ││              │└──────────┘                            │

    ││              │  ↓                                                │

    ││              │┌──────────┐                            │

    ││              ││신청사실의 공고 및  │    (보도자료, 인터넷 등)   │

    ││              ││의견수렴            │  ┌───────────┐│

    ││              ││                    │←│(필요시) 공청회       ││

    ││              ││                    │  └───────────┘│

    ││              ││                    │                            │

    ││              │└──────────┘                            │

    ││              │  ↓                                                │

    ││              │┌──────────┐                            │

    ││              ││예비인가 심사       │                            │

    ││              ││                    │  ┌───────────┐│

    ││              ││                    │←│(필요시) 실지조사     ││

    ││              ││                    │  └───────────┘│

    ││              ││                    │                            │

    ││              ││                    │  ┌───────────┐│

    ││              ││                    │←│(필요시) 평가위원회   ││

    ││              ││                    │  └───────────┘│

    ││              ││                    │                            │

    ││              │└──────────┘                            │

    ││              │  ↓                                                │

    ││              │┌──────────┐                            │

    ││              ││예비인가            │                            │

    │└───────┘└──────────┘                            │

    │  ?                ↓                                                │

    │                                            ┌───────────┐│

    │                                          ←│거부사실의 통보       ││

    │                                            └───────────┘│

    │                                                                      │

    │┌───────┐┌──────────┐                            │

    ││인가단계      ││인가 신청           │                            │

    ││              │└──────────┘                            │

    ││              │  ↓                                                │

    ││              │┌──────────┐  ┌───────────┐│

    ││              ││인가 심사ㆍ확인     │←│(필요시) 실지조사     ││

    ││              │└──────────┘  └───────────┘│

    ││              │                                                    │

    ││              │                          ┌───────────┐│

    ││              │  ↓                    ←│거부사실의 통보       ││

    ││              │                          └───────────┘│

    ││              │                                                    │

    ││              │┌──────────┐                            │

    ││              ││인가                │                            │

    │└───────┘└──────────┘                            │

    │                                                                      │

    └───────────────────────────────────┘'
  - 'Represent this sentence for retrieving relevant passages: 이 규정은 「금융위원회의 설치 등에
    관한 법률」(이하 "금융위설치법"이라 한다) 제17조제1호의 규정에 의하여 「상호저축은행법」(이하 "법"이라 한다)ㆍ같은 법 시행령(이하 "시행령"이라
    한다) 및 같은 법 시행규칙(이하 "시행규칙"이라 한다)과 「금융산업의 구조개선에 관한 법률」(이하 "금산법"이라 한다) 및 같은 법 시행령,
    그 밖에 법령에서 정하는 상호저축은행의 업무운용 및 감독에 관한 사항 중 금융위원회(이하 "금융위"라 한다) 소관사항의 시행에 필요한 사항을
    정함을 목적으로 한다.'
  - 'Represent this sentence for retrieving relevant passages: ① 법 법 제18조의2제1항제2호의
    업무용 부동산이라 함은 다음 각호의 1을 말한다. 1. 영업소(건물 연면적의 100분의 10이상을 업무에 직접 공여하는 경우) 2. 사택ㆍ기숙사ㆍ합숙소ㆍ연수원
    등의 용도로 직접 사용하는 부동산 ② 상호저축은행이 대주주 등과 부동산의 양도ㆍ양수계약을 체결하고자 하는 경우에는 감독원장의 승인을 받아야
    하며, 감독원장은 동 계약이 통상의 거래조건에 비해 해당 상호저축은행에 손실을 발생시키는지 여부를 심사하여야 한다.'
  - 'Represent this sentence for retrieving relevant passages: ① 상호저축은행은 법 제23조의2
    및 시행령 제13조의 규정에 의하여 결산일로부터 3월이내에 다음 각호에서 정하는 사항을 공시하여야 한다. 다만, 분기별 결산결과에 대한 공시자료는
    분기별 결산일로부터 2월 이내에 공시하여야 한다. 1. 시행령 제13조제1항제1호 내지 제3호에서 규정한 사항 및 내부통제에 관한 사항 2.
    건전성ㆍ수익성ㆍ생산성등을 나타내는 경영지표에 관한 사항 3. 대주주ㆍ임원과의 거래내역 및 대주주 발행주식 취득 현황 4. 제22조의3제1항
    각 호에 따른 업종별 신용공여의 규모, 연체율 및 자산건전성분류 현황 5. 경영방침, 리스크관리등 경영에 중요한 영향을 미치는 사항으로 감독원장이
    별도로 요구하는 사항 ② 상호저축은행은 다음 각호의 1에 해당되어 경영의 건전성을 크게 해치거나 해칠 우려가 있는 경우 관련내용을 공시하여야
    한다. 1. 여신 거래처별로 상호저축은행의 자기자본의 100분의 10을 초과하는 부실대출이 신규로 발생한 경우. 다만 그 금액이 5억원 이하인
    경우에는 그러하지 아니하다. 2. 금융기관검사및제재에관한규정에 의하여 감독원장이 정하고 있는 금융사고가 발생하여 상호저축은행의 자기자본의 100분의
    5 이상의 손실이 발생하였거나 발생이 예상되는 경우. 다만 그 금액이 2억원 이하인 경우와 감독원장이 사고내용을 조사하여 직접 발표하는 경우에는
    그러하지 아니하다. 3. 시행령 제13조제1항제5호에서 규정한 사항 4. 「자본시장과 금융투자업에 관한 법률」제9조제15항에 따른 주권상장법인이
    아닌 상호저축은행에 다음 각목의 1에 해당되는 사항이 발생하는 경우 가. 재무구조에 중대한 변경을 초래하는 사항 나. 상호저축은행 경영환경에
    중대한 변경을 초래하는 사항 다. 재산등에 대규모변동을 초래하는 사항 라. 채권채무관계에 중대한 변동을 초래하는 사항 마. 투자 및 출자관계에
    관한 사항 바. 손익구조변경에 관한 사항 사. 기타 상호저축은행 경영에 중대한 영향을 미칠 수 있는 사항 5. 법ㆍ시행령 또는 법 제35조의2제1항제4호에
    따른 금융관계법령을 위반함에 따라 과태료 또는 과징금을 부과 받은 경우 6. 「주식회사 등의 외부감사에 관한 법률」에 따라 외부감사인 지정을
    받은 경우 ③ 상호저축은행은 이용자의 권익을 보호하기 위하여 금융거래상의 계약조건 등을 정확하게 공시하여야 한다. ④ 제1항 내지 제3항에서
    정하는 사항에 대한 구체적인 공시항목 및 방법은 각각 중앙회회장이 정하는 상호저축은행통일경영공시기준 및 상호저축은행통일상품공시기준에 따른다.
    ⑤ 제2항의 규정에 따라 공시하는 경우 공시 전에 그 내용을 감독원장에게 보고하여야 한다.다만, 법 제10조의2제4항에 따라 감독원장에게 보고한
    경우에는 그러하지 아니하다. ⑥ 감독원장은 제1항 내지 제4항에서 정하는 공시사항에 대하여 허위공시하거나 중요한 사항을 누락하는 등 불성실하게
    공시하는 경우에는 당해 상호저축은행에 대해 정정공시 또는 재공시를 명령할 수 있다.'
  - 'Represent this sentence for retrieving relevant passages: ① 시행령 제9조의2제1항제3호 및
    동조 제2항제4호의 규정에 의한 신용공여 한도 초과사유 또는 시행령 제9조의3제3호의 규정에 의한 신용공여 한도 초과기간의 연장사유의 인정
    또는 승인을 받고자 하는 상호저축은행은 금융위에 그 인정 또는 승인을 신청하여야 한다. ② 상호저축은행은 시행령 제9조의2에서 정하는 사유로
    인하여 법 제12조제1항에서 제3항까지의 규정에 의한 한도를 초과하거나 시행령 제9조의3의 규정에 따라 한도의 초과기간을 연장하는 경우에는
    한도초과일 또는 한도초과기간 연장일로부터 15일 이내에 감독원장에게 보고하여야 한다. ③ 제1항 및 제2항에서 정하는 사항 이외에 인정신청
    절차, 한도초과보고 절차 기타 필요한 사항은 감독원장이 정한다.'
  - 'Represent this sentence for retrieving relevant passages: ① 상호저축은행은 법 제10조의2제4항
    각 호에 따른 사유가 발생하는 경우에는 사유발생일로부터 7일 이내에 감독원장 또는 중앙회회장이 정하는 바에 따라 감독원장 또는 중앙회회장에게
    이를 보고하여야 한다. 다만, 시행령 제7조제4항 각 호의 어느 하나 따른 사유가 발생하는 경우 지체 없이 감독원장에게 이를 보고하여야 한다.
    ② 시행령 제7조제4항제1호 본문에서 "금융위원회가 정하는 기준"이란 사유발생일 전월말 현재의 예금등 합계액 잔액의 100분의 1을 말한다.
    ③ 시행령 제7조제4항제1호 단서에서 "금융위원회가 정하여 고시하는 사유"란 다음 각 호의 어느 하나에 해당하는 경우를 말한다. 1. 전체
    예금등의 지급액 중 예금등의 계약만기에 따른 지급액이 100분의 70을 초과하는 경우 2. 전체 예금등의 지급액 중 1건의 지급액이 100분의
    50을 초과하는 경우 3. 그 밖에 이에 준하는 것으로서 감독원장이 인정하는 경우 ④ 시행령 제7조제4항제2호에서 "금융위원회가 정하여 고시하는
    경우"란 「금융기관 검사 및 제재에 관한 규정」에 따라 감독원장이 정하고 있는 금융사고가 발생하여 상호저축은행의 자기자본의 100분의 5 이상(회수예상가액을
    빼지 아니한 금액을 말한다.)의 손실이 발생하였거나 발생이 예상되는 경우를 말한다. 다만, 그 금액이 2억원 이하인 경우와 감독원장이 사고내용을
    조사하여 직접 발표하는 경우에는 그러하지 아니하다. <개정 2024. 9. 27.>

    제3장 업무감독 [본장신설 2000. 6. 23.]'
  - 'Represent this sentence for retrieving relevant passages: ① 금산법 제10조에 따라 금융위는
    상호저축은행이 다음 각 호의 어느 하나에 해당되는 경우에는 당해 상호저축은행에 대하여 필요한 조치를 이행하도록 권고하여야 한다. 1. 제44조제1항제1호의
    위험가중자산에 대한 자기자본비율이 제44조제1항제1호에 따른 비율미만인 경우 2. 제45조제2항 및 제4항의 규정에 의한 경영실태평가 결과
    종합평가등급이 3등급 이상으로서 자산건전성 또는 자본적정성 부문의 평가 등급을 4등급(취약) 이하로 판정받은 경우 3. 거액의 금융사고 또는
    부실채권의 발생으로 제1호 또는 제2호의 기준에 해당될 것이 명백하다고 판단되는 경우 ② 제1항에서 정하는 필요한 조치라 함은 다음 각호의
    일부 또는 전부에 해당하는 조치를 말한다. 1. 인력 및 조직운영의 개선 2. 경비절감 3. 영업소 관리의 효율화 4. 유형자산, 투자자산,
    무형자산 및 비업무용자산 투자, 신규업무영역에의 진출 및 신규출자의 제한 5. 부실자산의 처분 6. 자본금의 증액 또는 감액 7. 이익배당의
    제한 8. 특별대손충당금의 설정 ③ 금융위는 제1항에 의한 권고를 하는 경우 당해 상호저축은행 또는 관련 임원에 대하여 주의 또는 경고 조치를
    할 수 있다.'
  - 'Represent this sentence for retrieving relevant passages: ① 감독원장은 시행령 제26조제1항의
    규정에 의하여 금융위의 권한을 위탁받아 수행함에 있어 다음 각호의 어느 하나에 해당하는 사항에 대하여는 그 처리결과를 매분기별로 금융위에 보고하고
    중요사항에 대하여는 지체없이 보고하여야 한다. 1. 법 제10조의2제2항의 규정에 의한 시정ㆍ보완의 권고 2. 법 제17조의 규정에 의한 차입한도의
    예외에 관한 승인 3. 법 제22조제2항의 규정에 의한 감독명령(법 제23조의 규정에 의한 검사 또는 자료의 분석ㆍ평가결과 시정이 필요하다고
    인정되는 경우 이에 관한 감독명령에 한한다.) 4. 법 제24조제1항 각호(제4호를 제외한다)의 규정에 의한 행정처분 ② 감독원장은 이 법에
    따른 인가심사를 진행함에 있어서 각각의 신청서 접수일로부터 다음 각 호에서 정하는 기간(각각의 심사기간에서 제외하는 기간은 포함하지 아니한다)을
    경과한 인가의 심사 진행 상황 및 예상 심사 종료 시점을 금융위가 소집된 달에 마지막으로 개최되는 정례금융위원회에 매월 보고하여야 한다. 1.
    법 제6조제2항에 따른 인가(예비인가를 거치지 아니한 경우): 인가신청서 접수일로부터 3개월 2. 법 제6조제2항에 따른 인가(예비인가를 거친
    경우): 인가신청서 접수일로부터 1개월'
  - 'Represent this sentence for retrieving relevant passages: ① 금융위 또는 감독원장은 적기시정조치
    대상 상호저축은행이 다음 각 호의 어느 하나에 해당하는 경우에는 3개월 이내의 범위에서 기간을 정하여 그 조치를 유예할 수 있다. 다만, 불가피한
    사유가 있는 경우 1개월 이내의 범위에서 그 기간을 연장할 수 있다. 1. 적기시정조치 대상 상호저축은행이 경영개선계획에 따라 자본 확충,
    자산 매각 등을 통하여 단기간내에 금산법 제10조제2항에 따른 기준을 충족시킬 수 있다고 인정되는 경우. 이 경우 해당 기준 충족 여부를 판단함에
    있어서 적기시정조치 유예 여부가 예금보험기금의 손실을 줄일 수 있는지 여부를 감안하여야 한다. 2. 그 밖에 제1호 전단에 준하는 사유가 있다고
    인정되는 경우. 이 경우 제1호 후단을 준용한다. ② 제1항에 따라 적기시정조치를 유예한 경우 감독원장과 예금보험공사(제49조제6항 또는 제8항에
    따라 의견을 제출한 경우에 한정한다)는 적기시정조치 유예 결정일부터 1년이 경과한 후 지체 없이 해당 적기시정조치 유예 결과에 대한 평가보고서를
    각각 작성하여 금융위에 보고하여야 한다.'
- source_sentence: 'Represent this sentence for searching relevant passages: 주식회사의
    이사·감사가 회사와 체결한 약정에 따라 업무를 다른 이사 등에게 포괄적으로 위임하여 이사·감사로서의 실질적인 업무를 수행하지 않고 소극적인
    직무만을 수행한 경우, 이사·감사로서의 자격을 부정하거나 주주총회 결의에서 정한 보수청구권의 효력을 부정할 수 있는지 여부(원칙적 소극) /
    소극적으로 직무를 수행하는 이사·감사의 보수청구권 행사가 제한되는 '
  sentences:
  - 'Represent this sentence for retrieving relevant passages: 제273조(업무집행의 권리의무) 무한책임사원은
    정관에 다른 규정이 없는 때에는 각자가 회사의 업무를 집행할 권리와 의무가 있다.'
  - 'Represent this sentence for retrieving relevant passages: 2 회사와 이사의 관계는 「민법」의
    위임에 관한 규정을 준용한다.'
  - 'Represent this sentence for retrieving relevant passages: 1운송인은 자기 또는 사용인이 운송에
    관한 주의를 해태하지 아니하였음을 증명하지 아니하면 여객이 운송으로 인하여 받은 손해를 배상할 책임을 면하지 못한다.'
  - 'Represent this sentence for retrieving relevant passages: 4정관으로 제290조 각호의 사항을
    정한 때에는 이사는 이에 관한 조사를 하게 하기 위하여 검사인의 선임을 법원에 청구하여야 한다. 다만, 제299조의2의 경우에는 그러하지 아니하다.'
  - 'Represent this sentence for retrieving relevant passages: 2이사와 감사중 발기인이었던 자ᆞ현물출자자
    또는 회사성립후 양수할 재산의 계약당사자인 자는 제1항의 조사ᆞ보고에 참가하지 못한다.'
  - 'Represent this sentence for retrieving relevant passages: 제164조(동전-부득이한 사유가 있는
    경우) 부득이한 사유가 있는 경우에는 창고업자는 전조의 규정에 불구하고 언제든지 임치물을 반환할 수 있다.'
  - 'Represent this sentence for retrieving relevant passages: 3이사와 감사의 전원이 제2항에 해당하는
    때에는 이사는 공증인으로 하여금 제1항의 조사ᆞ보고를 하게 하여야 한다.'
  - 'Represent this sentence for retrieving relevant passages: 1 사채권자집회는 해당 종류의 사채
    총액(상환받은 금액은 제외한다)의 500분의 1 이상을 가진 사채권자 중에서 1명 또는 여러 명의 대표자를 선임하여 그 결의할 사항의 결정을
    위임할 수 있다. <개정 2011.4.14>'
  - 'Represent this sentence for retrieving relevant passages: 2손해배상의 액을 정함에는 법원은
    피해자와 그 가족의 정상을 참작하여야 한다.'
  - 'Represent this sentence for retrieving relevant passages: 1이사와 감사는 취임후 지체없이 회사의
    설립에 관한 모든 사항이 법령 또는 정관의 규정에 위반되지 아니하는지의 여부를 조사하여 발기인에게 보고하여야 한다.'
  - 'Represent this sentence for retrieving relevant passages: 제2절 여객운송'
- source_sentence: 'Represent this sentence for searching relevant passages: 자본시장과
    금융투자업에 관한 법률 제157조의 내용은 무엇인가?'
  sentences:
  - 'Represent this sentence for retrieving relevant passages: 7 집합투자재산을 보관ᆞ관리하는 신탁업자는
    그 집합투자기구의 집합투자재산에 관한 정보를 자기의 고유재산의 운용, 자기가 운용하는 집합투자재산의 운용 또는 자기가 판매하는 집합투자증권의
    판매를 위하여 이용하여서는 아니 된다. <신설 2009.2.3>'
  - 'Represent this sentence for retrieving relevant passages: 1 예탁대상증권등의 발행인은 새로
    증권등을 발행하는 경우 그 증권등의 종류, 그 밖에 총리령으로 정하는 사항을 예탁결제원에 지체 없이 통지하여야 한다. <개정 2008.2.29>'
  - 'Represent this sentence for retrieving relevant passages: 4 금융위원회는 증권금융회사의 직원이
    제1항 각 호(제6호를 제외한다)의 어느 하나에 해당하거나 별표 9 각 호의 어느 하나에 해당하는 경우에는 다음 각 호의 어느 하나에 해당하는
    조치를 그 증권금융회사에 요구할 수 있다. <개정 2008.2.29>'
  - 'Represent this sentence for retrieving relevant passages: 3 증권금융회사의 상근 임직원은 금융투자업자
    및 금융투자업관계기관(그 상근 임직원이 소속된 증권금융회사를 제외한다)과 자금의 공여, 손익의 분배, 그 밖에 영업에 관하여 대통령령으로 정하는
    특별한 이해관계를 가져서는 아니 된다.'
  - 'Represent this sentence for retrieving relevant passages: 4 금융위원회는 제1항에 따른 조사를
    함에 있어서 필요하다고 인정되는 경우에는 금융투자업자, 금융투자업관계기관 또는 거래소에 대통령령으로 정하는 방법에 따라 조사에 필요한 자료의
    제출을 요구할 수 있다. <개정 2008.2.29>'
  - 'Represent this sentence for retrieving relevant passages: 5 금융위원회는 제4항에 따라 선임된
    시장감시위원장이 직무수행에 부적합하다고 인정되는 경우로서 대통령령으로 정하는 경우에는 그 선임된 날부터 1개월 이내에 그 사유를 구체적으로
    밝혀 해임을 요구할 수 있다. 이 경우 해임 요구된 시장감시위원장의 직무는 정지되며, 거래소는 2개월 이내에 시장감시위원장을 새로 선임하여야
    한다. <개정 2008.2.29>'
  - 'Represent this sentence for retrieving relevant passages: 1 집합투자기구평가회사는 대통령령으로
    정하는 사항이 포함된 영업행위준칙을 제정하여야 한다.'
  - 'Represent this sentence for retrieving relevant passages: 3 제1항의 효력의 발생은 그 증권신고서의
    기재사항이 진실 또는 정확하다는 것을 인정하거나 정부에서 그 증권의 가치를 보증 또는 승인하는 효력을 가지지 아니한다.'
  - 'Represent this sentence for retrieving relevant passages: 3 거래계획 보고자는 그 거래계획에
    따라 특정증권등의 거래등을 하여야 한다. 다만, 거래 당시의 시장 상황 등을 반영하여 필요한 경우에 한정하여 거래금액의 100분의 30 이하의
    비율로서 대통령령으로 정하는 바에 따라 거래계획과 달리 거래등을 할 수 있다.'
  - 'Represent this sentence for retrieving relevant passages: 제157조(위임장용지 등의 공시)
    금융위원회와 거래소는 제152조에 따른 위임장용지 및 참고서류, 제155조에 따른 서면 및 제156조에 따른 정정내용을 그 접수일부터 3년간
    비치하고, 인터넷 홈페이지 등을 이용하여 공시하여야 한다. <개정 2008.2.29>'
  - 'Represent this sentence for retrieving relevant passages: 4 집합투자업자는 제2항에 따른 평가위원회가
    집합투자재산을 평가한 경우 그 평가명세를 지체 없이 그 집합투자재산을 보관ᆞ관리하는 신탁업자에게 통보하여야 한다.'
- source_sentence: 'Represent this sentence for searching relevant passages: 사정변경을
    이유로 계약을 해제하거나 해지할 수 있는 경우 및 여기에서 말하는 ‘사정’의 의미 / 계속적 계약에서 경제적 상황의 변화로 당사자에게 불이익이
    발생했다는 것만으로 계약을 해지할 수 있는지 여부(소극)에 대한 법적 판단은?'
  sentences:
  - 'Represent this sentence for retrieving relevant passages: 1부재자의 생사가 5년간 분명하지
    아니한 때에는 법원은 이해관계인이나 검사의 청구에 의하여 실종선고를 하여야 한다.'
  - 'Represent this sentence for retrieving relevant passages: 1수인이 공동의 불법행위로 타인에게
    손해를 가한 때에는 연대하여 그 손해를 배상할 책임이 있다.'
  - 'Represent this sentence for retrieving relevant passages: 2전지에 임한 자, 침몰한 선박 중에
    있던 자, 추락한 항공기 중에 있던 자 기타 사망의 원인이 될 위난을 당한 자의 생사가 전쟁종지후 또는 선박의 침몰, 항공기의 추락 기타 위난이
    종료한 후 1년간 분명하지 아니한 때에도 제1항과 같다. <개정 1984.4.10>'
  - 'Represent this sentence for retrieving relevant passages: 제553조(훼손 등으로 인한 해제권의
    소멸) 해제권자의 고의나 과실로 인하여 계약의 목적물이 현저히 훼손되거나 이를 반환할 수 없게 된 때 또는 가공이나 개조로 인하여 다른 종류의
    물건으로 변경된 때에는 해제권은 소멸한다.'
  - 'Represent this sentence for retrieving relevant passages: 1 후견인의 임무가 종료된 때에는
    후견인 또는 그 상속인은 1개월 내에 피후견인의 재산에 관한 계산을 하여야 한다. 다만, 정당한 사유가 있는 경우에는 법원의 허가를 받아 그
    기간을 연장할 수 있다.'
  - 'Represent this sentence for retrieving relevant passages: 제4관 후견의 종료 <신설 2011.3.7>'
  - 'Represent this sentence for retrieving relevant passages: 제63조(임시이사의 선임) 이사가
    없거나 결원이 있는 경우에 이로 인하여 손해가 생길 염려 있는 때에는 법원은 이해관계인이나 검사의 청구에 의하여 임시이사를 선임하여야 한다.'
  - 'Represent this sentence for retrieving relevant passages: 1권리의 행사와 의무의 이행은 신의에
    좇아 성실히 하여야 한다.'
  - 'Represent this sentence for retrieving relevant passages: 제125조(대리권수여의 표시에 의한
    표현대리) 제삼자에 대하여 타인에게 대리권을 수여함을 표시한 자는 그 대리권의 범위내에서 행한 그 타인과 그 제삼자간의 법률행위에 대하여 책임이
    있다. 그러나 제삼자가 대리권없음을 알았거나 알 수 있었을 때에는 그러하지 아니하다.'
  - 'Represent this sentence for retrieving relevant passages: 제471조(영수증소지자에 대한 변제)
    영수증을 소지한 자에 대한 변제는 그 소지자가 변제를 받을 권한이 없는 경우에도 효력이 있다. 그러나 변제자가 그 권한없음을 알았거나 알 수
    있었을 경우에는 그러하지 아니하다.'
  - 'Represent this sentence for retrieving relevant passages: 2 제1항의 계산은 후견감독인이 있는
    경우에는 그가 참여하지 아니하면 효력이 없다.'
- source_sentence: 'Represent this sentence for searching relevant passages: 상법 제808조의
    내용은 무엇인가?'
  sentences:
  - 'Represent this sentence for retrieving relevant passages: 3제2항의 규정에 의한 주식의 매수가액은
    주주와 회사간의 협의에 의하여 결정한다. <개정 2001.7.24>'
  - 'Represent this sentence for retrieving relevant passages: 제9절 해산'
  - 'Represent this sentence for retrieving relevant passages: 제830조(제3자가 선적인인 경우의
    통지ᆞ선적) 용선자 외의 제3자가 운송물을 선적할 경우에 선장이 그 제3자를 확실히 알 수 없거나 그 제3자가 운송물을 선적하지 아니한 때에는
    선장은 지체 없이 용선자에게 그 통지를 발송하여야 한다. 이 경우 선적기간 이내에 한하여 용선자가 운송물을 선적할 수 있다.'
  - 'Represent this sentence for retrieving relevant passages: 1회사의 발기인, 업무집행사원, 이사,
    집행임원, 감사위원회 위원, 감사 또는 제386조제2항, 제407조제1항, 제415조 또는 제567조의 직무대행자, 지배인 기타 회사영업에
    관한 어느 종류 또는 특정한 사항의 위임을 받은 사용인이 그 임무에 위배한 행위로써 재산상의 이익을 취하거나 제3자로 하여금 이를 취득하게
    하여 회사에 손해를 가한 때에는 10년 이하의 징역 또는 3천만원 이하의 벌금에 처한다. <개정 1984.4.10, 1995.12.29, 1999.12.31,
    2011.4.14>'
  - 'Represent this sentence for retrieving relevant passages: 1운송인은 제807조제1항에 따른
    금액의 지급을 받기 위하여 법원의 허가를 받아 운송물을 경매하여 우선변제를 받을 권리가 있다.'
  - 'Represent this sentence for retrieving relevant passages: 16. 제302조제2항, 제347조,
    제420조, 제420조의2, 제474조제2항 또는 제514조을 위반하여 주식청약서, 신주인수권증서 또는 사채청약서를 작성하지 아니하거나 이에
    적을 사항을 적지 아니하거나 또는 부실하게 적은 경우'
  - 'Represent this sentence for retrieving relevant passages: 제168조(준용규정) 제108조와
    제146조의 규정은 창고업자에 준용한다. <개정 1962.12.12>'
  - 'Represent this sentence for retrieving relevant passages: 2. 제345조제1항의 주식의 상환에
    관한 종류주식의 경우 외에 각 주주가 가진 주식 수에 따라 균등한 조건으로 취득하는 것으로서 대통령령으로 정하는 방법'
  - 'Represent this sentence for retrieving relevant passages: 1 이사가 고의 또는 과실로 법령
    또는 정관에 위반한 행위를 하거나 그 임무를 게을리한 경우에는 그 이사는 회사에 대하여 연대하여 손해를 배상할 책임이 있다. <개정 2011.4.14>'
  - 'Represent this sentence for retrieving relevant passages: 1 이사는 결산기마다 다음 각 호의
    서류와 그 부속명세서를 작성하여 이사회의 승인을 받아야 한다.'
  - 'Represent this sentence for retrieving relevant passages: 2어느 운송구간에서 손해가 발생하였는지
    불분명한 경우 또는 손해의 발생이 성질상 특정한 지역으로 한정되지 아니하는 경우에는 운송인은 운송거리가 가장 긴 구간에 적용되는 법에 따라
    책임을 진다. 다만, 운송거리가 같거나 가장 긴 구간을 정할 수 없는 경우에는 운임이 가장 비싼 구간에 적용되는 법에 따라 책임을 진다.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_accuracy@20
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_precision@20
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_recall@20
- cosine_ndcg@1
- cosine_ndcg@3
- cosine_ndcg@5
- cosine_ndcg@10
- cosine_ndcg@20
- cosine_mrr@1
- cosine_mrr@3
- cosine_mrr@5
- cosine_mrr@10
- cosine_mrr@20
- cosine_map@1
- cosine_map@3
- cosine_map@5
- cosine_map@10
- cosine_map@20
model-index:
- name: SentenceTransformer based on intfloat/multilingual-e5-small
  results:
  - task:
      type: suppress-progress-ir
      name: Suppress Progress IR
    dataset:
      name: val
      type: val
    metrics:
    - type: cosine_accuracy@1
      value: 0.05917159763313609
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.11735700197238659
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.14201183431952663
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.1873767258382643
      name: Cosine Accuracy@10
    - type: cosine_accuracy@20
      value: 0.23175542406311636
      name: Cosine Accuracy@20
    - type: cosine_precision@1
      value: 0.05917159763313609
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.039447731755424056
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.028796844181459568
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.01932938856015779
      name: Cosine Precision@10
    - type: cosine_precision@20
      value: 0.011932938856015778
      name: Cosine Precision@20
    - type: cosine_recall@1
      value: 0.05917159763313609
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.11285338593030901
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.13622616699539775
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.17781065088757397
      name: Cosine Recall@10
    - type: cosine_recall@20
      value: 0.2114234056541749
      name: Cosine Recall@20
    - type: cosine_ndcg@1
      value: 0.05917159763313609
      name: Cosine Ndcg@1
    - type: cosine_ndcg@3
      value: 0.09062040253624286
      name: Cosine Ndcg@3
    - type: cosine_ndcg@5
      value: 0.10014864478623443
      name: Cosine Ndcg@5
    - type: cosine_ndcg@10
      value: 0.11375405206843245
      name: Cosine Ndcg@10
    - type: cosine_ndcg@20
      value: 0.12273681154644399
      name: Cosine Ndcg@20
    - type: cosine_mrr@1
      value: 0.05917159763313609
      name: Cosine Mrr@1
    - type: cosine_mrr@3
      value: 0.0838264299802761
      name: Cosine Mrr@3
    - type: cosine_mrr@5
      value: 0.08939842209072978
      name: Cosine Mrr@5
    - type: cosine_mrr@10
      value: 0.0951754797908644
      name: Cosine Mrr@10
    - type: cosine_mrr@20
      value: 0.09811701337396561
      name: Cosine Mrr@20
    - type: cosine_map@1
      value: 0.05917159763313609
      name: Cosine Map@1
    - type: cosine_map@3
      value: 0.08212798597413981
      name: Cosine Map@3
    - type: cosine_map@5
      value: 0.08728824238439623
      name: Cosine Map@5
    - type: cosine_map@10
      value: 0.09270068250837481
      name: Cosine Map@10
    - type: cosine_map@20
      value: 0.09500975786263871
      name: Cosine Map@20
---

# SentenceTransformer based on intfloat/multilingual-e5-small

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) <!-- at revision c007d7ef6fd86656326059b28395a7a03a7c5846 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: PeftModelForFeatureExtraction 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Represent this sentence for searching relevant passages: 상법 제808조의 내용은 무엇인가?',
    'Represent this sentence for retrieving relevant passages: 1운송인은 제807조제1항에 따른 금액의 지급을 받기 위하여 법원의 허가를 받아 운송물을 경매하여 우선변제를 받을 권리가 있다.',
    'Represent this sentence for retrieving relevant passages: 16. 제302조제2항, 제347조, 제420조, 제420조의2, 제474조제2항 또는 제514조을 위반하여 주식청약서, 신주인수권증서 또는 사채청약서를 작성하지 아니하거나 이에 적을 사항을 적지 아니하거나 또는 부실하게 적은 경우',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Suppress Progress IR

* Dataset: `val`
* Evaluated with <code>lex_dpr.eval.SuppressProgressIREvaluator</code>

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.0592     |
| cosine_accuracy@3   | 0.1174     |
| cosine_accuracy@5   | 0.142      |
| cosine_accuracy@10  | 0.1874     |
| cosine_accuracy@20  | 0.2318     |
| cosine_precision@1  | 0.0592     |
| cosine_precision@3  | 0.0394     |
| cosine_precision@5  | 0.0288     |
| cosine_precision@10 | 0.0193     |
| cosine_precision@20 | 0.0119     |
| cosine_recall@1     | 0.0592     |
| cosine_recall@3     | 0.1129     |
| cosine_recall@5     | 0.1362     |
| cosine_recall@10    | 0.1778     |
| cosine_recall@20    | 0.2114     |
| cosine_ndcg@1       | 0.0592     |
| cosine_ndcg@3       | 0.0906     |
| cosine_ndcg@5       | 0.1001     |
| cosine_ndcg@10      | 0.1138     |
| **cosine_ndcg@20**  | **0.1227** |
| cosine_mrr@1        | 0.0592     |
| cosine_mrr@3        | 0.0838     |
| cosine_mrr@5        | 0.0894     |
| cosine_mrr@10       | 0.0952     |
| cosine_mrr@20       | 0.0981     |
| cosine_map@1        | 0.0592     |
| cosine_map@3        | 0.0821     |
| cosine_map@5        | 0.0873     |
| cosine_map@10       | 0.0927     |
| cosine_map@20       | 0.095      |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 13,091 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, <code>sentence_2</code>, <code>sentence_3</code>, <code>sentence_4</code>, <code>sentence_5</code>, <code>sentence_6</code>, <code>sentence_7</code>, <code>sentence_8</code>, <code>sentence_9</code>, <code>sentence_10</code>, and <code>sentence_11</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          | sentence_2                                                                          | sentence_3                                                                         | sentence_4                                                                          | sentence_5                                                                          | sentence_6                                                                          | sentence_7                                                                          | sentence_8                                                                          | sentence_9                                                                          | sentence_10                                                                         | sentence_11                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | string                                                                              | string                                                                             | string                                                                              | string                                                                              | string                                                                              | string                                                                              | string                                                                              | string                                                                              | string                                                                              | string                                                                              |
  | details | <ul><li>min: 21 tokens</li><li>mean: 69.3 tokens</li><li>max: 148 tokens</li></ul> | <ul><li>min: 18 tokens</li><li>mean: 65.77 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 18 tokens</li><li>mean: 70.05 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 18 tokens</li><li>mean: 66.8 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 67.91 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 18 tokens</li><li>mean: 65.93 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 18 tokens</li><li>mean: 67.09 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 65.38 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 16 tokens</li><li>mean: 67.44 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 16 tokens</li><li>mean: 67.36 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 16 tokens</li><li>mean: 65.67 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 64.88 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                | sentence_1                                                                                                                                                                                                   | sentence_2                                                                                                                                                                                      | sentence_3                                                                                                                                                                                                                | sentence_4                                                                                                                                                                   | sentence_5                                                                                                                                                                                                                                                                                    | sentence_6                                                                                                                                | sentence_7                                                                                                                                                                                                                                                                                                                                                   | sentence_8                                                                                                                                                     | sentence_9                                                                                                                                                                | sentence_10                                                                                                                                                                                                    | sentence_11                                                                                                                                                            |
  |:----------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Represent this sentence for searching relevant passages: 손해배상액 예정의 의미와 기능에 대한 법적 판단은?</code>        | <code>Represent this sentence for retrieving relevant passages: 3손해배상액의 예정은 이행의 청구나 계약의 해제에 영향을 미치지 아니한다.</code>                                                                                             | <code>Represent this sentence for retrieving relevant passages: 제282조(지상권의 양도, 임대) 지상권자는 타인에게 그 권리를 양도하거나 그 권리의 존속기간 내에서 그 토지를 임대할 수 있다.</code>                                                 | <code>Represent this sentence for retrieving relevant passages: 1점유자의 승계인은 자기의 점유만을 주장하거나 자기의 점유와 전점유자의 점유를 아울러 주장할 수 있다.</code>                                                                                          | <code>Represent this sentence for retrieving relevant passages: 2전점유자의 점유를 아울러 주장하는 경우에는 그 하자도 계승한다.</code>                                                                  | <code>Represent this sentence for retrieving relevant passages: 1사망이 정기금채무자의 책임있는 사유로 인한 때에는 법원은 정기금채권자 또는 그 상속인의 청구에 의하여 상당한 기간 채권의 존속을 선고할 수 있다.</code>                                                                                                                                     | <code>Represent this sentence for retrieving relevant passages: 2전항의 경우에도 제727조의 권리를 행사할 수 있다.</code>                                     | <code>Represent this sentence for retrieving relevant passages: 1전조의 규정에 의하여 대리인이 복대리인을 선임한 때에는 본인에게 대하여 그 선임감독에 관한 책임이 있다.</code>                                                                                                                                                                                                                           | <code>Represent this sentence for retrieving relevant passages: 2대리인이 본인의 지명에 의하여 복대리인을 선임한 경우에는 그 부적임 또는 불성실함을 알고 본인에게 대한 통지나 그 해임을 태만한 때가 아니면 책임이 없다.</code> | <code>Represent this sentence for retrieving relevant passages: 1타인의 동산에 가공한 때에는 그 물건의 소유권은 원재료의 소유자에게 속한다. 그러나 가공으로 인한 가액의 증가가 원재료의 가액보다 현저히 다액인 때에는 가공자의 소유로 한다.</code> | <code>Represent this sentence for retrieving relevant passages: 2가공자가 재료의 일부를 제공하였을 때에는 그 가액은 전항의 증가액에 가산한다.</code>                                                                                            | <code>Represent this sentence for retrieving relevant passages: 제744조(도의관념에 적합한 비채변제) 채무없는 자가 착오로 인하여 변제한 경우에 그 변제가 도의관념에 적합한 때에는 그 반환을 청구하지 못한다.</code>               |
  | <code>Represent this sentence for searching relevant passages: 자본시장과 금융투자업에 관한 법률 제377조의 내용은 무엇인가?</code> | <code>Represent this sentence for retrieving relevant passages: 1거래소는 정관으로 정하는 바에 따라 다음 각 호의 업무를 행한다. 다만, 제3호 및 제4호의 업무는 제378조에 따라 금융위원회로부터 청산기관 또는 결제기관으로 지정된 거래소로 한정한다. <개정 2013.5.28></code>              | <code>Represent this sentence for retrieving relevant passages: 1 투자회사등은 자기의 계산으로 자기가 발행한 집합투자증권을 취득하거나 질권의 목적으로 받지 못한다. 다만, 다음 각 호의 어느 하나에 해당하는 경우에는 자기의 계산으로 자기가 발행한 집합투자증권을 취득할 수 있다.</code> | <code>Represent this sentence for retrieving relevant passages: 3 제1항 또는 제2항을 위반하여 주식을 취득한 자는 그 주식에 대한 의결권을 행사할 수 없으며, 금융위원회는 제1항 또는 제2항을 위반하여 증권 또는 장내파생상품을 매매한 자에게 6개월 이내의 기간을 정하여 그 시정을 명할 수 있다. <개정 2008.2.29></code> | <code>Represent this sentence for retrieving relevant passages: 6. 청산대상업자가 아닌 자가 청산대상업자를 통하여 금융투자상품거래청산회사로 하여금 청산대상거래의 채무를 부담하게 하는 경우 그 금융투자상품거래청산의 중개ᆞ주선이나 대리에 관한 사항</code> | <code>Represent this sentence for retrieving relevant passages: 2 사업보고서 제출대상법인은 제1항의 사업보고서에 다음 각 호의 사항을 기재하고, 대통령령으로 정하는 서류를 첨부하여야 한다. <개정 2013.5.28, 2016.3.29></code>                                                                                                                       | <code>Represent this sentence for retrieving relevant passages: 6 금융위원회는 제2항에 따라 인가를 한 경우에는 다음 각 호의 사항을 관보 및 인터넷 홈페이지 등에 공고하여야 한다.</code> | <code>Represent this sentence for retrieving relevant passages: 2 투자신탁을 설정한 집합투자업자는 제1항에 따른 청구가 있는 경우 해당 수익자에게 수익증권의 매수에 따른 수수료, 그 밖의 비용을 부담시켜서는 아니 된다.</code>                                                                                                                                                                                               | <code>Represent this sentence for retrieving relevant passages: 1. 거래소시장에서 이상거래의 혐의가 있다고 인정되는 해당 증권의 종목 또는 장내파생상품 매매 품목의 거래상황을 파악하기 위한 경우</code>               | <code>Represent this sentence for retrieving relevant passages: 1 투자회사등은 투자회사등의 업무와 관련한 자료를 대통령령으로 정하는 자료의 종류별로 대통령령으로 정하는 기간 동안 기록ᆞ유지하여야 한다.</code>                      | <code>Represent this sentence for retrieving relevant passages: 2 금융위원회는 제1항의 허가를 하고자 하는 경우에는 다음 각 호의 사항을 심사하여야 한다. <개정 2008.2.29, 2013.5.28></code>                                                           | <code>Represent this sentence for retrieving relevant passages: 1 제103조제1항제4호부터 제6호까지의 어느 하나에 규정된 재산만을 수탁받는 신탁업자가 관리형신탁계약을 체결하는 경우 그 신탁재산에 수반되는 금전채권을 수탁할 수 있다.</code> |
  | <code>Represent this sentence for searching relevant passages: 자본시장과 금융투자업에 관한 법률 제418조의 내용은 무엇인가?</code> | <code>Represent this sentence for retrieving relevant passages: 제418조(보고사항) 금융투자업자(겸영금융투자업자의 경우에는 제6호부터 제9호까지에 한한다)는 다음 각 호의 어느 하나에 해당하는 경우에는 대통령령으로 정하는 방법에 따라 그 사실을 금융위원회에 보고하여야 한다. <개정 2008.2.29></code> | <code>Represent this sentence for retrieving relevant passages: 4 공개매수자가 제1항 또는 제3항에 따라 공개매수신고서의 정정신고서를 제출하는 경우 공개매수기간의 종료일은 다음 각 호와 같다.</code>                                                 | <code>Represent this sentence for retrieving relevant passages: 2 투자신탁을 설정한 집합투자업자는 제1항에 따른 청구가 있는 경우 해당 수익자에게 수익증권의 매수에 따른 수수료, 그 밖의 비용을 부담시켜서는 아니 된다.</code>                                                            | <code>Represent this sentence for retrieving relevant passages: 5 제4항의 검토기간을 산정할 때 등록신청서 흠결의 보완기간 등 총리령으로 정하는 기간은 검토기간에 산입하지 아니한다.</code>                                    | <code>Represent this sentence for retrieving relevant passages: 1 제195조는 투자유한회사의 정관변경에 관하여 준용한다. 이 경우 "투자회사"는 각각 "투자유한회사"로, 같은 조 제1항 중 "이사회 결의로"는 "법인이사가"로, "제201조제2항 단서"는 "제210조제2항 단서"로, 같은 조 제1항 중 "주주총회의 결의" 및 같은 조 제2항 중 "이사회 결의 및 주주총회의 결의"는 각각 "사원총회의 결의"로, "주주"는 각각 "사원"으로 본다.</code> | <code>Represent this sentence for retrieving relevant passages: 4. 그 밖에 투자자 보호 또는 건전한 거래질서를 위하여 필요한 사항으로서 대통령령으로 정하는 사항</code>            | <code>Represent this sentence for retrieving relevant passages: 1 증권선물위원회는 제172조부터 제174조까지, 제176조, 제178조, 제178조의2, 제180조 및 제180조의2부터 제180조의5까지의 규정을 위반한 행위(이하 이 조에서 "위반행위"라 한다)를 조사하기 위하여 필요하다고 인정되는 경우에는 금융위원회 소속공무원 중 대통령령으로 정하는 자(이하 이 조에서 "조사공무원"이라 한다)에게 위반행위의 혐의가 있는 자를 심문하거나 물건을 압수 또는 사업장 등을 수색하게 할 수 있다. <개정 2008.2.29, 2016.3.29, 2021.1.5></code> | <code>Represent this sentence for retrieving relevant passages: 7. 그 밖에 위법행위를 시정하거나 방지하기 위하여 필요한 조치로서 대통령령으로 정하는 조치</code>                                     | <code>Represent this sentence for retrieving relevant passages: 4. 「행정기본법」 제29조 각 호의 사유가 해소되어 과징금을 한꺼번에 납부할 수 있다고 인정되는 경우</code>                                          | <code>Represent this sentence for retrieving relevant passages: 8 일반 사모집합투자업자는 등록 이후 그 영업을 영위하는 경우 제2항 각 호의 등록요건(같은 항 제6호는 제외하며, 같은 항 제2호 및 제5호의 경우에는 대통령령으로 정하는 완화된 요건을 말한다)을 유지하여야 한다. <개정 2021.4.20></code> | <code>Represent this sentence for retrieving relevant passages: 2. 신탁재산에 속하는 주식을 발행한 법인이 자기주식을 확보하기 위하여 신탁계약에 따라 신탁업자에게 취득하게 한 그 법인의 주식</code>                         |
* Loss: <code>lex_dpr.trainer.losses.MixedNegativesRankingLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `num_train_epochs`: 200
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 200
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch   | Step | Training Loss | val loss | val cosine loss | val_cosine_ndcg@20 |
|:-------:|:----:|:-------------:|:--------:|:---------------:|:------------------:|
| 1.0     | 205  | -             | 2.5747   | 2.5747          | 0.0501             |
| 1.4634  | 300  | -             | 2.5775   | 2.5775          | 0.0502             |
| 2.0     | 410  | -             | 2.5731   | 2.5731          | 0.0503             |
| 2.4390  | 500  | 6.5531        | -        | -               | -                  |
| 2.9268  | 600  | -             | 2.5718   | 2.5718          | 0.0518             |
| 3.0     | 615  | -             | 2.5721   | 2.5721          | 0.0525             |
| 4.0     | 820  | -             | 2.5679   | 2.5679          | 0.0533             |
| 4.3902  | 900  | -             | 2.5684   | 2.5684          | 0.0540             |
| 4.8780  | 1000 | 6.553         | -        | -               | -                  |
| 5.0     | 1025 | -             | 2.5617   | 2.5617          | 0.0578             |
| 5.8537  | 1200 | -             | 2.5571   | 2.5571          | 0.0609             |
| 6.0     | 1230 | -             | 2.5553   | 2.5553          | 0.0612             |
| 7.0     | 1435 | -             | 2.5462   | 2.5462          | 0.0670             |
| 7.3171  | 1500 | 6.5517        | 2.5460   | 2.5460          | 0.0687             |
| 8.0     | 1640 | -             | 2.5352   | 2.5352          | 0.0723             |
| 8.7805  | 1800 | -             | 2.5256   | 2.5256          | 0.0770             |
| 9.0     | 1845 | -             | 2.5239   | 2.5239          | 0.0782             |
| 9.7561  | 2000 | 6.5528        | -        | -               | -                  |
| 10.0    | 2050 | -             | 2.5105   | 2.5105          | 0.0849             |
| 10.2439 | 2100 | -             | 2.5118   | 2.5118          | 0.0866             |
| 11.0    | 2255 | -             | 2.5018   | 2.5018          | 0.0918             |
| 11.7073 | 2400 | -             | 2.4916   | 2.4916          | 0.0957             |
| 12.0    | 2460 | -             | 2.4894   | 2.4894          | 0.0992             |
| 12.1951 | 2500 | 6.5513        | -        | -               | -                  |
| 13.0    | 2665 | -             | 2.4769   | 2.4769          | 0.1051             |
| 13.1707 | 2700 | -             | 2.4740   | 2.4740          | 0.1057             |
| 14.0    | 2870 | -             | 2.4575   | 2.4575          | 0.1077             |
| 14.6341 | 3000 | 6.5523        | 2.4495   | 2.4495          | 0.1096             |
| 15.0    | 3075 | -             | 2.4470   | 2.4470          | 0.1106             |
| 16.0    | 3280 | -             | 2.4311   | 2.4311          | 0.1137             |
| 16.0976 | 3300 | -             | 2.4318   | 2.4318          | 0.1138             |
| 17.0    | 3485 | -             | 2.4146   | 2.4146          | 0.1150             |
| 17.0732 | 3500 | 6.5509        | -        | -               | -                  |
| 17.5610 | 3600 | -             | 2.4060   | 2.4060          | 0.1153             |
| 18.0    | 3690 | -             | 2.3987   | 2.3987          | 0.1159             |
| 19.0    | 3895 | -             | 2.3811   | 2.3811          | 0.1168             |
| 19.0244 | 3900 | -             | 2.3827   | 2.3827          | 0.1169             |
| 19.5122 | 4000 | 6.5518        | -        | -               | -                  |
| 20.0    | 4100 | -             | 2.3621   | 2.3621          | 0.1171             |
| 20.4878 | 4200 | -             | 2.3529   | 2.3529          | 0.1182             |
| 21.0    | 4305 | -             | 2.3496   | 2.3496          | 0.1194             |
| 21.9512 | 4500 | 6.5515        | 2.3327   | 2.3327          | 0.1199             |
| 22.0    | 4510 | -             | 2.3276   | 2.3276          | 0.1202             |
| 23.0    | 4715 | -             | 2.3108   | 2.3108          | 0.1215             |
| 23.4146 | 4800 | -             | 2.3026   | 2.3026          | 0.1219             |
| 24.0    | 4920 | -             | 2.2967   | 2.2967          | 0.1219             |
| 24.3902 | 5000 | 6.5501        | -        | -               | -                  |
| 24.8780 | 5100 | -             | 2.2749   | 2.2749          | 0.1225             |
| 25.0    | 5125 | -             | 2.2674   | 2.2674          | 0.1223             |
| 26.0    | 5330 | -             | 2.2535   | 2.2535          | 0.1231             |
| 26.3415 | 5400 | -             | 2.2437   | 2.2437          | 0.1221             |
| 26.8293 | 5500 | 6.5509        | -        | -               | -                  |
| 27.0    | 5535 | -             | 2.2331   | 2.2331          | 0.1224             |
| 27.8049 | 5700 | -             | 2.2101   | 2.2101          | 0.1232             |
| 28.0    | 5740 | -             | 2.2104   | 2.2104          | 0.1230             |
| 29.0    | 5945 | -             | -        | -               | 0.1227             |


### Framework Versions
- Python: 3.12.12
- Sentence Transformers: 3.4.1
- Transformers: 4.43.4
- PyTorch: 2.5.1+cu121
- Accelerate: 0.34.2
- Datasets: 2.21.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->