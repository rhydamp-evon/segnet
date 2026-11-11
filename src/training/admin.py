from django.contrib import admin
from training.models import TrainingSession
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.http import HttpResponseRedirect

from training.tasks import run_buddy_pipeline, evaluate_model
from atlentis_ml_api.settings import APP_ENV


class TrainingAdmin(admin.ModelAdmin):
    change_list_template = 'admin/training_session_list_template.html'
    change_form_template = 'admin/training_session_detail_template.html'
    add_form_template = 'admin/training_session_add_template.html'
    date_heirarchy = (
        'created_at',
    )
    list_display_links = ('id', 'name')

    def get_list_display(self, request, extra_context=None):
        list_display = ['id', 'name', 'learning_rate', 'batch_size',
                        'num_of_epochs', 'status', 'more_actions']
        if not request.user.has_perm('training.start_training'):
            if 'more_actions' in list_display:
                list_display.remove('more_actions')
        return list_display

    def more_actions(self, obj):
        acns = '''
            <button id="start-training" onclick="startTraining({});" style="background-color: #4CAF50;border: none;color: white; border-radius:15px;cursor: pointer;" >start</button>&nbsp; \
            <button id="stop-training" onclick="stopTraining({});" style="background-color: #f44336;border: none;color: white;border-radius:15px;cursor: pointer;" >stop</button>&nbsp;
            '''
        # acns = '''
        #     <button id="start-training" onclick="startTraining({});" style="background-color: #4CAF50;border: none;color: white; border-radius:15px;cursor: pointer;" >start</button>
        #     <a href="/api/v1/training/stop/" >Stop</a>&nbsp;
        #     <a href="/api/v1/training/pause/" >Pause</a>&nbsp;
        #     '''
        return format_html(mark_safe(acns), obj.id, obj.id)

    def change_view(self, request, object_id, form_url='', extra_context=None):
        extra_context = {"train_session_id": object_id}
        return super(TrainingAdmin, self).change_view(
            request, object_id, form_url, extra_context=extra_context,
        )

    def add_view(self, request, form_url='', extra_context=None):
        return super(TrainingAdmin, self).add_view(request, form_url='', extra_context=None)

    def response_change(self, request, obj):
        if "deploy_to_sbx" in request.POST:
            run_buddy_pipeline.delay(obj.name)
            self.message_user(request, "Started model deployment")
            return HttpResponseRedirect(".")
        if "evaluate_model" in request.POST:
            evaluate_model.delay(obj.name)
            self.message_user(request, "Started model evaluation")
            return HttpResponseRedirect(".")
        return super().response_change(request, obj)

    more_actions.show_change_link = True
    more_actions.allow_tags = True


if APP_ENV != "production":
    admin.site.register(TrainingSession, TrainingAdmin)
