from PySide6.QtCore import QThread, Signal
import base64
import json
import os

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

class GenAIWorker(QThread):
    finished_signal = Signal(str, str) # status, output_path
    error_signal = Signal(str)

    def __init__(self, api_key, prompt, image_path=None, style_preset="", aspect_ratio="1:1"):
        super().__init__()
        self.api_key = api_key
        self.prompt = prompt
        self.image_path = image_path
        self.style_preset = style_preset
        self.aspect_ratio = aspect_ratio

    def run(self):
        if not genai:
            self.error_signal.emit("google-genai 모듈이 설치되어 있지 않습니다.")
            return
            
        try:
            client = genai.Client(api_key=self.api_key)
            full_prompt = f"{self.prompt}, {self.style_preset}, high quality"
            
            use_fallback = False
            
            # Google GenAI 최신 표준 API (generate_content + IMAGE modality) 시도
            try:
                from PIL import Image
                
                contents = [full_prompt]
                if self.image_path and os.path.exists(self.image_path):
                    try:
                        pil_img = Image.open(self.image_path)
                        contents.append(pil_img)
                    except Exception as img_err:
                        print(f"[GenAIWorker] 이미지 로드 경고: {img_err}")

                response = client.models.generate_content(
                    model='gemini-2.5-flash-image',
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=['IMAGE'],
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    )
                )

                img_data = None
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if getattr(part, 'inline_data', None) and part.inline_data.data:
                            img_data = part.inline_data.data
                            break

                if img_data:
                    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rendered_output.png")
                    with open(output_path, "wb") as f:
                        f.write(img_data)
                    self.finished_signal.emit("success", output_path)
                    return
                else:
                    use_fallback = True
            except Exception as e:
                # 무료 티어 할당량 초과(Quota 0) 또는 지원되지 않는 키일 경우 폴백 실행
                print(f"[GenAIWorker] Gemini Image API 호출 건너뜀 (폴백 전환): {e}")
                use_fallback = True

            if use_fallback:
                # 무료 공용 이미지 생성 API (Pollinations) 폴백
                import urllib.request
                import urllib.parse

                safe_prompt = urllib.parse.quote(full_prompt)
                fallback_url = f"https://image.pollinations.ai/prompt/{safe_prompt}?nologo=true&width=4677&height=2631"

                output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rendered_output.png")
                req = urllib.request.Request(fallback_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as response, open(output_path, 'wb') as out_file:
                    out_file.write(response.read())

                self.finished_signal.emit("success", output_path)
                
        except Exception as e:
            self.error_signal.emit(f"AI 렌더링 중 오류 발생: {str(e)}")


class MaterialAIWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, prompt, api_key=None):
        super().__init__()
        self.prompt = prompt
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def run(self):
        if not self.api_key:
            self.error.emit("GEMINI_API_KEY가 설정되어 있지 않습니다.")
            return

        if not genai:
            self.error.emit("google-genai 모듈이 설치되어 있지 않습니다.")
            return

        try:
            client = genai.Client(api_key=self.api_key)
            system_prompt = (
                "당신은 3D 그래픽스 및 PBR(Physically Based Rendering) 머티리얼 전문가입니다.\n"
                "사용자가 제공한 재질/질감 설명을 분석하여 3D 렌더링에 필요한 PBR 파라미터를 도출하고,\n"
                "반드시 아래 JSON 스키마 규격의 순수 JSON 데이터만 반환하세요.\n"
                "{\n"
                '  "desc": "재질에 대한 1줄 요약",\n'
                '  "color": [0.8, 0.8, 0.8], // 0.0 ~ 1.0 범위의 알베도 기본 색상 float 리스트\n'
                '  "bright": 1.0, // 0.0 ~ 10.0 범위의 float (선명도/밝기 계수, 기본 1.0)\n'
                '  "roughness": 0.5, // 0.0 ~ 1.0 범위의 float (표면 거칠기: 매끄러울수록 0, 거칠수록 1)\n'
                '  "metallic": 0.0, // 0.0 ~ 1.0 범위의 float (금속성: 비금속 0, 완전 금속 1)\n'
                '  "reflection": 0.5, // 0.0 ~ 1.0 범위의 float (반사율/스펙큘러 강도)\n'
                '  "refraction": 1.5, // 1.0 ~ 3.0 범위의 float (굴절률 IOR: 공기 1.0, 물 1.33, 유리/플라스틱 1.5)\n'
                '  "normal_scale": 1.0, // 0.0 ~ 3.0 범위의 float (노멀맵/요철 돌출 강도)\n'
                '  "emissive": 0.0, // 0.0 ~ 5.0 범위의 float (자가 발광 강도)\n'
                '  "coat_strength": 0.0, // 0.0 ~ 1.0 범위의 float (투명 코팅층 반사 강도)\n'
                '  "coat_roughness": 0.0, // 0.0 ~ 1.0 범위의 float (코팅층 표면 거칠기)\n'
                '  "anisotropy": 0.0, // 0.0 ~ 1.0 범위의 float (비등방성 결 무늬 반사)\n'
                '  "occlusion": 1.0 // 0.0 ~ 5.0 범위의 float (구석진 틈새 AO 음영 강도)\n'
                "}"
            )
            user_msg = f"재질 느낌/설명: {self.prompt}"

            resp = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[system_prompt, user_msg],
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    temperature=0.2
                )
            )

            raw_text = resp.text.strip()
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()

            data = json.loads(raw_text)
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(f"AI 분석 중 오류 발생: {str(e)}")

