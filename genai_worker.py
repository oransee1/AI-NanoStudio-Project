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
