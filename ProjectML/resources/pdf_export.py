from fpdf import FPDF

title = 'Evaluation Report'

class PDF(FPDF):
    def __init__(self,imgdata):
        self.imgdata=imgdata

    def decode(self):
        imgdt = base64.b64decode(str(imgdata))
        self.img = Image.open(io.BytesIO(imgdt))

    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Calculate width of title and position
        w = self.get_string_width(title) + 6
        self.set_x((210 - w) / 2)
        # Colors of frame, background and text
        # self.set_draw_color(0, 80, 180)
        # self.set_fill_color(230, 230, 0)
        self.set_text_color(220, 50, 50)
        # Thickness of frame (1 mm)
        # self.set_line_width(1)
        # Title
        self.cell(w, 9, title, 1, 1, 'C', 1)
        # Line break
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Text color in gray
        self.set_text_color(128)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def chapter_title(self, name):
        # Arial 12
        self.set_font('Arial', '', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, 'Plot: %s' % (name), 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def chapter_body(self):
        self.decode()
        self.image(self.img)
        self.ln()

    def print_chapter(self, num, title, encstr):
        self.add_page()
        self.chapter_title(num, title)
        self.chapter_body()

class EvalReport():
    def __init__(self, enc_str_dict):
        self.enc_str_dict = enc_str_dict

    def classification_report():
        pass

    def regression_report():
        pass
# pdf = PDF()
# pdf.set_title(title)
# pdf.print_chapter(1, 'A RUNAWAY REEF', 'test_img.py')
# pdf.print_chapter(2, 'THE PROS AND CONS', 'lin_regression.py')
# pdf.output('tuto3.pdf', 'F')
