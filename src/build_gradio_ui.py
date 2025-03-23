from system_utils import *
import gradio as gr

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'>Medical AI Assistant (Chat Mode)</h1>")

        # conversation state: a list of (user, assistant) for gr.Chatbot
        conv_state = gr.State([])

        with gr.Row():
            # ---------------------------
            # Left column: Chat interface
            # ---------------------------
            with gr.Column(scale=1):
                chatbox = gr.Chatbot(
                    label="Chat Conversation",
                    height=600
                )
                new_query_input = gr.Textbox(
                    label="New Query",
                    placeholder="Type your question or message here...",
                    lines=2
                )
                send_query_btn = gr.Button("Send")

            # ---------------------------
            # Right column: Other functionalities
            # ---------------------------
            with gr.Column(scale=1):
                gr.Markdown("### Initial Report Generation")
                audio_upload = gr.Audio(
                    label="Upload Audio File",
                    type="filepath"
                )
                init_manual_input = gr.Textbox(
                    label="Initial Supplementary Input",
                    lines=3,
                    placeholder="Enter initial supplementary note here..."
                )
                generate_report_btn = gr.Button("Generate Medical Report")

                status_box = gr.Textbox(
                    label="Status",
                    interactive=False
                )

                gr.Markdown("### Archive / Reset")
                with gr.Row():
                    patient_id_input = gr.Textbox(
                        label="Patient ID",
                        placeholder="e.g., patient_001"
                    )
                    archive_status_box = gr.Textbox(
                        label="Archive Status",
                        interactive=False
                    )
                with gr.Row():
                    archive_btn = gr.Button("Archive Conversation")
                    reset_btn = gr.Button("New Query (Reset)")

        # ---------------------------------------------------
        # BINDING SECTION
        # ---------------------------------------------------

        # 1) Generate Medical Report
        #   chat_list: a list of (user_msg, assistant_msg)
        #   status: a string
        generate_report_btn.click(
            fn=update_dialogue_generator,
            inputs=[init_manual_input, audio_upload, conv_state],
            outputs=[conv_state, status_box],
            queue=True
        ).then(
            fn=lambda c_list: c_list,
            inputs=conv_state,
            outputs=chatbox
        )

        # 2) Send question (chat_with_assistant)
        #   chat_with_assistant returns (chat_list, status)
        send_query_btn.click(
            fn=chat_with_assistant,
            inputs=[new_query_input, conv_state],
            outputs=[conv_state, status_box],
            queue=True
        ).then(
            fn=lambda c_list: c_list,
            inputs=conv_state,
            outputs=chatbox
        )

        # 3) Archive conversation
        archive_btn.click(
            fn=archive_current,
            inputs=[patient_id_input, conv_state],
            outputs=archive_status_box
        )

        # 4) Reset
        reset_btn.click(
            fn=reset_all,
            inputs=[],
            outputs=[chatbox, status_box, archive_status_box, conv_state]
        )

    return demo