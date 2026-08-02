"""Structured screening for disabled/dementia-prone older adults.

This module supports research-oriented screening and environmental/assistive-device
evaluation. It never produces a clinical diagnosis. Every result includes risk
stratification, actionable recommendations and escalation guidance.
"""

from typing import Any, Dict, List

DISCLAIMER = (
    "本结果仅用于科研筛查、环境改造与康复辅助器具适配参考，不构成疾病诊断、"
    "伤残等级认定或治疗建议；高风险结果应由老年医学、康复医学等专业人员进一步评估。"
)

ADL_ITEMS = ("bathing", "dressing", "toileting", "transferring", "continence", "feeding")
ENVIRONMENT_WEIGHTS = {
    "entrance_steps_without_ramp": 3,
    "bathroom_no_grab_bar": 3,
    "bathroom_slippery": 3,
    "poor_night_lighting": 2,
    "loose_rugs_or_wires": 2,
    "bed_height_inappropriate": 2,
    "doorway_too_narrow": 2,
    "no_emergency_call": 2,
    "kitchen_fire_risk": 3,
    "medication_unmanaged": 3,
    "wandering_exit_unprotected": 3,
    "toilet_inaccessible": 3,
}


def assessment_catalog() -> Dict[str, Any]:
    return {
        "adl": {
            "name": "日常生活活动能力（ADL）筛查",
            "items": list(ADL_ITEMS),
            "answer": "每项填写 1（可独立完成）或 0（需要帮助/不能完成）",
        },
        "cognition": {
            "name": "认知功能风险初筛",
            "items": ["orientation", "delayed_recall", "clock_drawing"],
            "answer": "定向力 0-5、延迟回忆 0-3、画钟 0-2",
        },
        "environment": {
            "name": "老年人居住环境安全评价",
            "items": list(ENVIRONMENT_WEIGHTS),
            "answer": "存在风险填 true，不存在填 false",
        },
        "assistive_device": {
            "name": "康复辅助器具适配初评",
            "items": [
                "mobility", "transfer", "upper_limb", "cognition_risk",
                "fall_history", "living_alone",
            ],
            "answer": "按老年人的真实功能状态填写",
        },
    }


def assess(kind: str, answers: Dict[str, Any]) -> Dict[str, Any]:
    handlers = {
        "adl": _assess_adl,
        "cognition": _assess_cognition,
        "environment": _assess_environment,
        "assistive_device": _assess_assistive_device,
    }
    if kind not in handlers:
        raise ValueError("assessment_type must be adl, cognition, environment or assistive_device")
    result = handlers[kind](answers or {})
    result.update({"assessment_type": kind, "disclaimer": DISCLAIMER})
    return result


def _bounded_int(value: Any, low: int, high: int, field: str) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} 必须为 {low}-{high} 的整数")
    if not low <= value <= high:
        raise ValueError(f"{field} 必须为 {low}-{high} 的整数")
    return value


def _assess_adl(a: Dict[str, Any]) -> Dict[str, Any]:
    missing = [x for x in ADL_ITEMS if x not in a]
    if missing:
        raise ValueError("缺少 ADL 项目: " + ", ".join(missing))
    score = sum(_bounded_int(a[x], 0, 1, x) for x in ADL_ITEMS)
    if score == 6:
        level, risk = "基本独立", "low"
    elif score >= 4:
        level, risk = "轻度依赖", "medium"
    elif score >= 2:
        level, risk = "中度依赖", "high"
    else:
        level, risk = "重度依赖", "critical"
    dependent = [x for x in ADL_ITEMS if int(a[x]) == 0]
    recs = ["对依赖项目开展照护需求与康复目标复评"]
    if "transferring" in dependent or "toileting" in dependent:
        recs.append("重点评估移位、如厕通道及扶手/移位辅具需求")
    if score <= 3:
        recs.append("建议由康复治疗师评估照护等级和辅助器具适配")
    return {"score": score, "max_score": 6, "level": level, "risk_level": risk,
            "dependent_items": dependent, "recommendations": recs}


def _assess_cognition(a: Dict[str, Any]) -> Dict[str, Any]:
    orientation = _bounded_int(a.get("orientation"), 0, 5, "orientation")
    recall = _bounded_int(a.get("delayed_recall"), 0, 3, "delayed_recall")
    clock = _bounded_int(a.get("clock_drawing"), 0, 2, "clock_drawing")
    score = orientation + recall + clock
    if score >= 8:
        level, risk = "暂未见明显认知风险", "low"
    elif score >= 5:
        level, risk = "存在认知下降风险", "medium"
    else:
        level, risk = "认知损害高风险", "high"
    recs = ["结合教育程度、听视力及情绪状态解释结果"]
    if risk != "low":
        recs.extend(["建议接受标准化认知量表及老年医学专科评估", "排查谵妄、抑郁、药物等可逆因素"])
    if risk == "high":
        recs.append("加强走失、用火、服药和财务安全管理")
    return {"score": score, "max_score": 10, "level": level, "risk_level": risk,
            "recommendations": recs}


def _assess_environment(a: Dict[str, Any]) -> Dict[str, Any]:
    hazards: List[Dict[str, Any]] = []
    for key, weight in ENVIRONMENT_WEIGHTS.items():
        if bool(a.get(key, False)):
            hazards.append({"item": key, "weight": weight})
    score = sum(x["weight"] for x in hazards)
    if score <= 3:
        level, risk = "低风险", "low"
    elif score <= 9:
        level, risk = "中风险", "medium"
    else:
        level, risk = "高风险", "high"
    recs = []
    names = {x["item"] for x in hazards}
    if names & {"bathroom_no_grab_bar", "bathroom_slippery", "toilet_inaccessible"}:
        recs.append("优先进行卫生间防滑、扶手、坐便与通行空间改造")
    if names & {"poor_night_lighting", "loose_rugs_or_wires", "bed_height_inappropriate"}:
        recs.append("改善夜间照明并清除跌倒障碍物，调整床椅高度")
    if names & {"wandering_exit_unprotected", "kitchen_fire_risk", "medication_unmanaged"}:
        recs.append("配置走失/烟火/服药监测与照护者告警机制")
    if not recs:
        recs.append("维持定期巡检，并结合功能变化动态复评")
    return {"score": score, "max_score": sum(ENVIRONMENT_WEIGHTS.values()),
            "level": level, "risk_level": risk, "hazards": hazards,
            "recommendations": recs}


def _assess_assistive_device(a: Dict[str, Any]) -> Dict[str, Any]:
    mobility = str(a.get("mobility", "independent"))
    transfer = str(a.get("transfer", "independent"))
    upper_limb = str(a.get("upper_limb", "normal"))
    cognition = str(a.get("cognition_risk", "low"))
    devices: List[Dict[str, str]] = []
    if mobility == "mild_support":
        devices.append({"device": "手杖/四脚杖", "reason": "轻度步行支持与稳定性需求"})
    elif mobility == "moderate_support":
        devices.append({"device": "助行器", "reason": "需要双侧支撑并降低跌倒风险"})
    elif mobility == "wheelchair":
        devices.append({"device": "个性化轮椅及减压坐垫", "reason": "步行受限，需评估座宽、姿势与压力管理"})
    if transfer != "independent":
        devices.append({"device": "床边扶手/移位板/移位机", "reason": "降低移位过程中的跌倒与照护者损伤"})
    if upper_limb != "normal":
        devices.append({"device": "加粗柄进食及穿衣辅具", "reason": "上肢抓握或精细活动受限"})
    if bool(a.get("fall_history")):
        devices.append({"device": "髋部保护与跌倒报警设备", "reason": "既往跌倒提示再次跌倒高风险"})
    if cognition in ("medium", "high"):
        devices.append({"device": "定位/离床/服药提醒设备", "reason": "认知风险下的走失、离床和服药安全需求"})
    if bool(a.get("living_alone")):
        devices.append({"device": "一键呼叫与远程照护终端", "reason": "独居场景需要紧急联络和异常告警"})
    if not devices:
        devices.append({"device": "暂不建议固定辅具", "reason": "当前信息未显示明确适配需求，建议定期复评"})
    return {"level": "需专业适配复核" if len(devices) > 1 else "基础建议",
            "risk_level": "high" if cognition == "high" or transfer == "dependent" else "medium",
            "recommended_devices": devices,
            "recommendations": ["辅具选型前应完成身体尺寸、使用环境、照护者能力和试用效果评估",
                                "避免仅依据疾病名称购买辅具，使用后应持续随访适配效果"]}
