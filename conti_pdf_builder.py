import os
from datetime import datetime
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

class ContiPDFBuilder:
    """
    ReportLab 기반 시네마틱 콘티 기획서 PDF 자동 생성기.
    """

    @staticmethod
    def build(start_img_path: str, end_img_path: str, motion_prompt: str,
              scene_desc: str, camera_data: dict, output_path: str) -> str:
        
        # A4 가로 방향
        width, height = landscape(A4)
        c = canvas.Canvas(output_path, pagesize=(width, height))
        
        # 폰트 설정
        font_name = "Helvetica"
        font_bold = "Helvetica-Bold"
        malgun_path = r"C:\Windows\Fonts\malgun.ttf"
        
        try:
            if os.path.exists(malgun_path):
                pdfmetrics.registerFont(TTFont('Malgun', malgun_path))
                font_name = "Malgun"
                font_bold = "Malgun"
        except Exception:
            pass

        # 배경 및 테두리 (옵션)
        c.setLineWidth(1)
        c.rect(1*cm, 1*cm, width - 2*cm, height - 2*cm)

        # 1. 상단 헤더
        c.setFont(font_bold, 20)
        c.drawString(1.5*cm, height - 2.5*cm, "NanoStudio-3D Cinematic Scene Sheet")
        c.setFont(font_name, 12)
        c.drawRightString(width - 1.5*cm, height - 2.5*cm, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 선 그리기
        c.line(1.5*cm, height - 2.8*cm, width - 1.5*cm, height - 2.8*cm)

        # 2. 시작/종료 프레임 이미지 (가로 배치)
        img_y = height - 11.5*cm
        img_width, img_height = 10*cm, 7.5*cm
        
        # 시작 이미지
        if os.path.exists(start_img_path):
            c.drawImage(start_img_path, 1.5*cm, img_y, width=img_width, height=img_height, preserveAspectRatio=True)
        else:
            c.setFillColorRGB(0.9, 0.9, 0.9)
            c.rect(1.5*cm, img_y, img_width, img_height, fill=1)
            c.setFillColorRGB(0.5, 0.5, 0.5)
            c.drawCentredString(1.5*cm + img_width/2, img_y + img_height/2, "No Image")
        
        c.setFillColorRGB(0, 0, 0)
        c.setFont(font_bold, 12)
        c.drawCentredString(1.5*cm + img_width/2, img_y - 0.7*cm, "Shot A (Start)")

        # 종료 이미지
        if os.path.exists(end_img_path):
            c.drawImage(end_img_path, 13*cm, img_y, width=img_width, height=img_height, preserveAspectRatio=True)
        else:
            c.setFillColorRGB(0.9, 0.9, 0.9)
            c.rect(13*cm, img_y, img_width, img_height, fill=1)
            c.setFillColorRGB(0.5, 0.5, 0.5)
            c.drawCentredString(13*cm + img_width/2, img_y + img_height/2, "No Image")
        
        c.setFillColorRGB(0, 0, 0)
        c.setFont(font_bold, 12)
        c.drawCentredString(13*cm + img_width/2, img_y - 0.7*cm, "Shot B (End)")

        # 3. 카메라 좌표 테이블
        start_cam = camera_data.get('start', {})
        end_cam = camera_data.get('end', {})
        delta_cam = camera_data.get('delta', {})

        def fmt_tuple(t):
            if not t: return "-"
            return f"({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})"

        data = [
            ['Parameter', 'Start (A)', 'End (B)'],
            ['Position', fmt_tuple(start_cam.get('position')), fmt_tuple(end_cam.get('position'))],
            ['Focal Point', fmt_tuple(start_cam.get('focal_point')), fmt_tuple(end_cam.get('focal_point'))],
            ['View Up', fmt_tuple(start_cam.get('view_up')), fmt_tuple(end_cam.get('view_up'))]
        ]
        
        table = Table(data, colWidths=[3.5*cm, 5.5*cm, 5.5*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,-1), font_name),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        
        table.wrapOn(c, width, height)
        table.drawOn(c, 1.5*cm, 4.5*cm)

        # 4. 이동 벡터 요약
        c.setFont(font_bold, 12)
        c.drawString(17*cm, 8.5*cm, "Movement Delta:")
        c.setFont(font_name, 10)
        delta_str = (f"dx: {delta_cam.get('dx', 0):.2f} | dy: {delta_cam.get('dy', 0):.2f} | dz: {delta_cam.get('dz', 0):.2f}\n"
                     f"Distance: {delta_cam.get('distance', 0):.2f} units\n"
                     f"Yaw Angle: {delta_cam.get('yaw_deg', 0):.2f}°")
        
        dy = 8.0*cm
        for line in delta_str.split('\n'):
            c.drawString(17*cm, dy, line)
            dy -= 0.5*cm

        # 5. 씬 콘셉트 텍스트
        c.setFont(font_bold, 12)
        c.drawString(1.5*cm, 3.5*cm, "Scene Concept:")
        c.setFont(font_name, 10)
        c.drawString(1.5*cm, 3.0*cm, scene_desc[:120] + ("..." if len(scene_desc) > 120 else ""))

        # 6. Veo 모션 프롬프트 (박스 테두리)
        c.setFont(font_bold, 12)
        c.drawString(11*cm, 3.5*cm, "Veo Motion Prompt:")
        c.setLineWidth(1)
        c.rect(11*cm, 1.8*cm, 17*cm, 1.5*cm)
        c.setFont(font_name, 10)
        c.drawString(11.2*cm, 2.5*cm, motion_prompt[:100] + ("..." if len(motion_prompt) > 100 else ""))

        # 7. 하단 푸터
        c.setFont(font_name, 9)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawRightString(width - 1.5*cm, 1.2*cm, "Synthesized by Gemini / Nano Banana AI — NanoStudio-3D")

        c.save()
        return output_path

if __name__ == '__main__':
    dummy_camera = {
        "start": {"position": (10.0, 5.0, 5.0), "focal_point": (0.0, 0.0, 0.0), "view_up": (0.0, 1.0, 0.0)},
        "end": {"position": (5.0, 5.0, 10.0), "focal_point": (0.0, 0.0, 0.0), "view_up": (0.0, 1.0, 0.0)},
        "delta": {"dx": -5.0, "dy": 0.0, "dz": 5.0, "distance": 7.07, "yaw_deg": 45.0}
    }
    output = "sample_conti.pdf"
    
    # 더미 이미지 경로 (존재하지 않으므로 회색 박스 출력됨)
    ContiPDFBuilder.build(
        start_img_path="dummy_start.jpg",
        end_img_path="dummy_end.jpg",
        motion_prompt="A slow panning shot moving from right to left, revealing the futuristic cityscape.",
        scene_desc="사이버펑크 도시의 아침, 네온사인이 서서히 꺼지고 태양이 떠오르는 장면.",
        camera_data=dummy_camera,
        output_path=output
    )
    print(f"Sample PDF created at: {output}")
