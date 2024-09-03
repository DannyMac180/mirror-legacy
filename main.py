from fasthtml.common import *

app,rt = fast_app()

@rt('/')
def get(): return Textarea(id='prompt-textarea', tabindex='0', data_id='root', dir='auto', rows='1', placeholder='Message ChatGPT', style='height: 40px; overflow-y: hidden;', cls='m-0 resize-none border-0 bg-transparent px-0 text-token-text-primary focus:ring-0 focus-visible:ring-0 max-h-[25dvh] max-h-52')

serve()