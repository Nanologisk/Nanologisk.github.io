—
layout: archive
title: Archive
permalink: /archive/
—

<h2>Tags from DataScience Collection</h2>

{% assign all_tags = «» | split: «» %}
{% assign tag_map = {} %}

{% for page in site.datascience %}
  {% for tag in page.tags %}
    {% assign all_tags = all_tags | push: tag %}
    {% capture tag_slug %}{{ tag | slugify }}{% endcapture %}
    {% if tag_map[tag_slug] %}
      {% assign existing = tag_map[tag_slug] %}
      {% assign updated = existing | push: page %}
      {% assign tag_map = tag_map | merge: tag_slug: updated %}
    {% else %}
      {% assign tag_map = tag_map | merge: tag_slug: [page] %}
    {% endif %}
  {% endfor %}
{% endfor %}

<ul>
  {% assign all_tags = all_tags | uniq | sort %}
  {% for tag in all_tags %}
    <li><a href=«#{{ tag | slugify }}»>{{ tag }} ({{ tag_map[tag | slugify].size }})</a></li>
  {% endfor %}
</ul>

{% for tag in all_tags %}
  <h3 id=«{{ tag | slugify }}»>{{ tag }}</h3>
  <ul>
    {% assign pages = tag_map[tag | slugify] %}
    {% for page in pages %}
      <li><a href=«{{ page.url }}»>{{ page.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}