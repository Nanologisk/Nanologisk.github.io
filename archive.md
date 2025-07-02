---
layout: archive
title: Archive
permalink: /archive/
---

<h2>Tags from DataScience Collection</h2>

<ul>
  {% assign all_tags = site.datascience | map: ‘tags’ | compact | flatten | uniq | sort %}
  {% for tag in all_tags %}
    <li><a href=«#{{ tag | slugify }}»>{{ tag }}</a></li>
  {% endfor %}
</ul>

{% for tag in all_tags %}
  <h3 id=«{{ tag | slugify }}»>{{ tag }}</h3>
  <ul>
    {% for page in site.datascience %}
      {% if page.tags contains tag %}
        <li><a href=«{{ page.url }}»>{{ page.title }}</a></li>
      {% endif %}
    {% endfor %}
  </ul>
{% endfor %}