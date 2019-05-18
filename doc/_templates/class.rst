{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% if methods %}

   {% for item in methods %}
   {%- if item not in inherited_members %}
   .. automethod:: {{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
