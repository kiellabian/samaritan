from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

from views import Home, ClassifyView

urlpatterns = patterns('',
    # Examples:
    url(r'^$', Home.as_view(), name='home'),
    url(r'^classifiy/$', ClassifyView.as_view(), name='classify'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
)
