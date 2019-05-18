{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}

   {% for item in methods %}
   {% if item in ['forward','call','initialize'] %}
   .. automethod:: {{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
